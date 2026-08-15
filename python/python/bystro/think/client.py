"""Synchronous, event-driven client for Bystro Think workloads."""

# ``Run`` is intentionally a friend handle over ``ThinkClient`` internals.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import mimetypes
from pathlib import Path
import threading
import time
from types import TracebackType
from typing import Protocol, Self, TypeAlias, cast
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
import uuid

import requests
import socketio
from socketio.exceptions import SocketIOError

from bystro.api import auth as dashboard_auth
from bystro.api.auth import CachedAuth, LegalConsent
from bystro.think.context import (
    MessageInput,
    MessageWithContext,
    PreviousConversation,
)
from bystro.think.errors import (
    InputResponseError,
    RunProtocolError,
    RunRejectedError,
    RunTimeoutError,
    ThinkAuthenticationError,
    ThinkBillingRequiredError,
    ThinkConnectionError,
    ThinkError,
    ThinkHTTPError,
)
from bystro.think.models import (
    Dataset,
    EventKind,
    FileInput,
    InputKind,
    NeedsInput,
    RunOptions,
    RunOutcome,
    RunResult,
    RunStatus,
    ThinkEvent,
    ThinkMessage,
    UploadedFile,
    UploadPhase,
    UploadProgress,
)

logger = logging.getLogger(__name__)

DEFAULT_THINK_URL = "https://ai.bystro.cloud"
DEFAULT_UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024
_DEFAULT_HTTP_TIMEOUT = (15.0, 300.0)
_DEFAULT_SOCKET_TIMEOUT = 90.0
_DEFAULT_FINALIZATION_TIMEOUT = 10.0
_CHECKPOINT_REPLAY_INTERVAL = 0.25
_ARTIFACT_PATH_MAX_LENGTH = 2_048
_ARTIFACT_PATH_MAX_DEPTH = 64
_MAX_EVENT_HISTORY = 2_000

ProgressCallback: TypeAlias = Callable[[ThinkEvent], None]
UploadProgressCallback: TypeAlias = Callable[[UploadProgress], None]


class _Response(Protocol):
    status_code: int
    reason: str

    def json(self) -> object: ...


class _CookieJar(Protocol):
    def set(self, name: str, value: str, **kwargs: object) -> object: ...


class _HTTPSession(Protocol):
    @property
    def cookies(self) -> _CookieJar: ...

    def post(self, url: str, **kwargs: object) -> _Response: ...

    def get(self, url: str, **kwargs: object) -> _Response: ...

    def close(self) -> None: ...


class _SocketClient(Protocol):
    connected: bool

    def on(
        self,
        event: str,
        handler: Callable[..., object] | None = None,
        namespace: str | None = None,
    ) -> object: ...

    def connect(self, url: str, **kwargs: object) -> None: ...

    def call(
        self,
        event: str,
        data: object = None,
        timeout: float | None = None,
    ) -> object: ...

    def emit(self, event: str, data: object = None) -> None: ...

    def disconnect(self) -> None: ...


SocketFactory: TypeAlias = Callable[..., _SocketClient]


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    raw_mapping = cast(Mapping[object, object], value)
    return {key: item for key, item in raw_mapping.items() if isinstance(key, str)}


def _boolean(value: object) -> bool:
    return value is True


def _normalize_base_url(url: str) -> str:
    normalized = dashboard_auth.normalize_url(url)
    parsed = urlsplit(normalized)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _normalize_artifact_path(
    artifact_path: str | None,
    original_filename: str,
) -> str | None:
    """Preflight the public artifact-path contract enforced by Think."""

    if artifact_path is None or not artifact_path.strip():
        return None
    normalized = artifact_path.strip().replace("\\", "/")
    if len(normalized) > _ARTIFACT_PATH_MAX_LENGTH:
        raise ValueError("Artifact path is too long")
    if normalized.startswith("/"):
        raise ValueError("Artifact path must be relative")
    parts = [part.strip() for part in normalized.split("/")]
    if len(parts) > _ARTIFACT_PATH_MAX_DEPTH or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Artifact path contains invalid components")
    original_leaf = original_filename.replace("\\", "/").split("/")[-1]
    if parts[-1] != original_leaf:
        raise ValueError("Artifact path must end with the uploaded file name")
    if parts[0] == ".metadata":
        raise ValueError("Artifact path uses a reserved folder")
    return "/".join(parts)


def _json_payload(response: _Response) -> dict[str, object]:
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise ThinkHTTPError(
            "Think returned an invalid JSON response",
            status_code=response.status_code,
        ) from exc
    if not isinstance(payload, Mapping):
        raise ThinkHTTPError(
            "Think returned an invalid JSON response",
            status_code=response.status_code,
        )
    return _mapping(cast(Mapping[object, object], payload))


def _nested_error_fields(payload: Mapping[str, object]) -> tuple[str | None, str | None]:
    code = _text(payload.get("error")) or _text(payload.get("code")) or _text(payload.get("status"))
    message = _text(payload.get("message"))
    detail = payload.get("detail")
    if isinstance(detail, str):
        message = detail.strip() or message
    elif isinstance(detail, Mapping):
        nested_code, nested_message = _nested_error_fields(
            _mapping(cast(Mapping[object, object], detail))
        )
        code = code or nested_code
        message = message or nested_message
    return code, message


def _raise_for_response(response: _Response, *, action: str) -> dict[str, object]:
    try:
        payload = _json_payload(response)
    except ThinkHTTPError:
        if 200 <= response.status_code < 300:
            raise
        payload = {}
    if 200 <= response.status_code < 300:
        return payload
    code, server_message = _nested_error_fields(payload)
    message = server_message or f"Think {action} failed ({response.status_code} {response.reason})"
    if response.status_code == 401:
        raise ThinkAuthenticationError(
            message,
            status_code=response.status_code,
            code=code,
            retryable=False,
        )
    if response.status_code == 402 or code == "billing_required":
        raise ThinkBillingRequiredError(
            message,
            status_code=response.status_code,
            code=code,
            retryable=False,
        )
    raise ThinkHTTPError(
        message,
        status_code=response.status_code,
        code=code,
        retryable=response.status_code in {408, 429} or response.status_code >= 500,
    )


def _thread_id(payload: Mapping[str, object]) -> str | None:
    return _text(payload.get("threadId")) or _text(payload.get("id"))


def _message_from_payload(payload: Mapping[str, object], run_id: str) -> ThinkMessage | None:
    message_id = _text(payload.get("id"))
    message_type = _text(payload.get("type"))
    if message_id is None or message_type is None:
        return None
    output_value = payload.get("output")
    output = output_value if isinstance(output_value, str) else str(output_value or "")
    return ThinkMessage(
        id=message_id,
        run_id=run_id,
        type=message_type,
        output=output,
        name=_text(payload.get("name")),
        metadata=_mapping(payload.get("metadata")),
    )


def _uploaded_file(payload: Mapping[str, object]) -> UploadedFile:
    file_id = _text(payload.get("id"))
    name = _text(payload.get("name"))
    if file_id is None or name is None:
        raise RunProtocolError("Think upload response omitted the file id or name")
    display_name = _text(payload.get("displayName")) or name
    raw_size = payload.get("size")
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise RunProtocolError("Think upload response contained an invalid file size")
    return UploadedFile(
        id=file_id,
        name=name,
        display_name=display_name,
        size=raw_size,
        mime=_text(payload.get("mime")),
    )


@dataclass(slots=True)
class _RunTracker:
    run_id: str | None
    options: RunOptions
    condition: threading.Condition = field(default_factory=threading.Condition)
    status: RunStatus = RunStatus.SUBMITTING
    events: list[ThinkEvent] = field(default_factory=list)
    messages: dict[str, ThinkMessage] = field(default_factory=dict)
    message_order: list[str] = field(default_factory=list)
    needs_input: NeedsInput | None = None
    final_output: str | None = None
    failure: Exception | None = None
    sequence: int = 0
    latest_checkpoint_id: str | None = None
    answered_checkpoint_id: str | None = None
    final_message_id: str | None = None
    completed_message_id: str | None = None
    task_ended_at: float | None = None
    refresh_requested: bool = False
    awaiting_transcript: bool = False


class ThinkClient:
    """A synchronous client for one active Think conversation at a time.

    Constructing the client is offline. The first upload, submission, or
    explicit :meth:`connect` call validates the cached dashboard JWT with the
    Think service and opens its live event transport.
    """

    def __init__(
        self,
        auth: CachedAuth | None = None,
        *,
        think_url: str = DEFAULT_THINK_URL,
        on_event: ProgressCallback | None = None,
        upload_chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
        upload_max_retries: int = 8,
        upload_finalize_timeout: float = 24 * 60 * 60,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        http_timeout: tuple[float, float] = _DEFAULT_HTTP_TIMEOUT,
        finalization_timeout: float = _DEFAULT_FINALIZATION_TIMEOUT,
        transports: Iterable[str] | None = None,
        _session: _HTTPSession | None = None,
        _socket_factory: SocketFactory | None = None,
        _sleep: Callable[[float], None] = time.sleep,
        _clock: Callable[[], float] = time.monotonic,
    ) -> None:
        resolved_auth = auth if auth is not None else dashboard_auth.load_state()
        if resolved_auth is None:
            raise ThinkAuthenticationError(
                "No cached Bystro login was found. Call bystro.api.auth.login first.",
                status_code=401,
                code="not_logged_in",
            )
        if upload_chunk_size <= 0:
            raise ValueError("upload_chunk_size must be positive")
        if upload_max_retries < 0:
            raise ValueError("upload_max_retries cannot be negative")
        if upload_finalize_timeout <= 0:
            raise ValueError("upload_finalize_timeout must be positive")
        if finalization_timeout <= 0:
            raise ValueError("finalization_timeout must be positive")

        self.auth = resolved_auth
        self.think_url = _normalize_base_url(think_url)
        self.session_id = str(uuid.uuid4())
        self._on_event = on_event
        self._upload_chunk_size = upload_chunk_size
        self._upload_max_retries = upload_max_retries
        self._upload_finalize_timeout = upload_finalize_timeout
        self._socket_timeout = socket_timeout
        self._http_timeout = http_timeout
        self._finalization_timeout = finalization_timeout
        self._transports = tuple(transports) if transports is not None else None
        self._sleep = _sleep
        self._clock = _clock
        self._session = (
            _session
            if _session is not None
            else cast(_HTTPSession, requests.Session())
        )
        self._owns_session = _session is None
        factory = _socket_factory or cast(SocketFactory, socketio.Client)
        self._socket = factory(
            http_session=self._session,
            reconnection=True,
            reconnection_attempts=0,
            handle_sigint=False,
        )
        self._state_lock = threading.RLock()
        self._connect_lock = threading.Lock()
        self._active_tracker: _RunTracker | None = None
        self._bootstrapped = False
        self._closed = False
        self._register_socket_handlers()

    @classmethod
    def login(
        cls,
        email: str,
        password: str,
        *,
        dashboard_url: str = dashboard_auth.DEFAULT_BYSTRO_URL,
        think_url: str = DEFAULT_THINK_URL,
        cache: bool = True,
        legal_consent: LegalConsent | None = None,
        site_access_code: str | None = None,
        on_event: ProgressCallback | None = None,
        upload_chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
        upload_max_retries: int = 8,
        upload_finalize_timeout: float = 24 * 60 * 60,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        http_timeout: tuple[float, float] = _DEFAULT_HTTP_TIMEOUT,
        finalization_timeout: float = _DEFAULT_FINALIZATION_TIMEOUT,
        transports: Iterable[str] | None = None,
    ) -> Self:
        """Log in through the dashboard and construct a Think client."""

        state = dashboard_auth.login(
            email,
            password,
            host=dashboard_url,
            cache=cache,
            site_access_code=site_access_code,
            legal_consent=legal_consent,
        )
        return cls(
            auth=state,
            think_url=think_url,
            on_event=on_event,
            upload_chunk_size=upload_chunk_size,
            upload_max_retries=upload_max_retries,
            upload_finalize_timeout=upload_finalize_timeout,
            socket_timeout=socket_timeout,
            http_timeout=http_timeout,
            finalization_timeout=finalization_timeout,
            transports=transports,
        )

    @classmethod
    def from_cached_login(
        cls,
        *,
        think_url: str = DEFAULT_THINK_URL,
        on_event: ProgressCallback | None = None,
        upload_chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
        upload_max_retries: int = 8,
        upload_finalize_timeout: float = 24 * 60 * 60,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        http_timeout: tuple[float, float] = _DEFAULT_HTTP_TIMEOUT,
        finalization_timeout: float = _DEFAULT_FINALIZATION_TIMEOUT,
        transports: Iterable[str] | None = None,
    ) -> Self:
        """Construct a client from ``~/.bystro`` authentication state."""

        return cls(
            think_url=think_url,
            on_event=on_event,
            upload_chunk_size=upload_chunk_size,
            upload_max_retries=upload_max_retries,
            upload_finalize_timeout=upload_finalize_timeout,
            socket_timeout=socket_timeout,
            http_timeout=http_timeout,
            finalization_timeout=finalization_timeout,
            transports=transports,
        )

    @property
    def connected(self) -> bool:
        """Whether the live Socket.IO transport is connected."""

        return bool(self._socket.connected)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _register_socket_handlers(self) -> None:
        handlers: dict[str, Callable[..., object]] = {
            "connect": self._handle_connect,
            "connect_error": self._handle_connect_error,
            "disconnect": self._handle_disconnect,
            "task_start": self._handle_task_start,
            "task_end": self._handle_task_end,
            "new_message": self._handle_message,
            "update_message": self._handle_message,
            "stream_start": self._handle_message,
            "stream_token": self._handle_stream_token,
            "status_overlay_update": self._handle_overlay,
            "resume_thread": self._handle_resume_thread,
            "resume_thread_error": self._handle_resume_error,
            "toast": self._handle_toast,
        }
        for event, handler in handlers.items():
            self._socket.on(event, handler=handler)

    def _socket_origin_and_path(self) -> tuple[str, str]:
        parsed = urlsplit(self.think_url)
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        root = parsed.path.rstrip("/")
        socket_path = f"{root}/ws/socket.io".lstrip("/")
        return origin, socket_path

    def _active_thread_id(self) -> str:
        with self._state_lock:
            tracker = self._active_tracker
        if tracker is None:
            return ""
        with tracker.condition:
            return tracker.run_id or ""

    def _socket_auth(self) -> dict[str, str]:
        return {
            "clientType": "webapp",
            "sessionId": self.session_id,
            "threadId": self._active_thread_id(),
            "userEnv": json.dumps({}, separators=(",", ":")),
            "chatProfile": "",
        }

    def _bootstrap_http_session(self) -> None:
        parsed = urlsplit(self.think_url)
        hostname = parsed.hostname
        if hostname is None:
            raise ThinkConnectionError("Think URL has no hostname")
        self._session.cookies.set(
            "bystro_access_token",
            self.auth.access_token,
            domain=hostname,
            path="/",
            secure=parsed.scheme == "https",
        )
        try:
            auth_response = self._session.post(
                f"{self.think_url}/auth/cookie",
                json={},
                timeout=self._http_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ThinkConnectionError("Could not reach Think authentication") from exc
        _raise_for_response(auth_response, action="authentication")
        try:
            sticky_response = self._session.post(
                f"{self.think_url}/set-session-cookie",
                json={"session_id": self.session_id},
                timeout=self._http_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ThinkConnectionError("Could not establish Think session routing") from exc
        _raise_for_response(sticky_response, action="session setup")
        self._bootstrapped = True

    def connect(self) -> None:
        """Authenticate with Think and open the live event connection."""

        with self._connect_lock:
            if self._closed:
                raise ThinkConnectionError("Think client is closed")
            if self._socket.connected:
                return
            if not self._bootstrapped:
                self._bootstrap_http_session()
            origin, socket_path = self._socket_origin_and_path()
            kwargs: dict[str, object] = {
                "socketio_path": socket_path,
                "auth": self._socket_auth,
                "wait_timeout": self._socket_timeout,
            }
            if self._transports is not None:
                kwargs["transports"] = list(self._transports)
            try:
                self._socket.connect(origin, **kwargs)
            except (SocketIOError, OSError) as exc:
                raise ThinkConnectionError("Could not open the Think live connection") from exc

    def _handle_connect(self) -> None:
        self._socket.emit("connection_successful")
        tracker = self._active_tracker_snapshot()
        if tracker is not None:
            self._record_event(tracker, EventKind.CONNECTED, message="Connected to Think")
            with tracker.condition:
                run_id = tracker.run_id
            if run_id:
                self._emit_status_ready(run_id)

    def _handle_connect_error(self, error: object) -> None:
        tracker = self._active_tracker_snapshot()
        if tracker is not None:
            self._record_event(
                tracker,
                EventKind.DISCONNECTED,
                message=f"Think connection error: {error}",
            )

    def _handle_disconnect(self, reason: object = None) -> None:
        tracker = self._active_tracker_snapshot()
        if tracker is not None:
            self._record_event(
                tracker,
                EventKind.DISCONNECTED,
                message=f"Disconnected from Think: {reason or 'unknown reason'}",
            )

    def _active_tracker_snapshot(self) -> _RunTracker | None:
        with self._state_lock:
            return self._active_tracker

    def _tracker_for_payload(self, payload: Mapping[str, object]) -> _RunTracker | None:
        incoming_thread_id = _thread_id(payload)
        if incoming_thread_id is None:
            return None
        tracker = self._active_tracker_snapshot()
        if tracker is None:
            return None
        with tracker.condition:
            if tracker.run_id is None:
                tracker.run_id = incoming_thread_id
            elif tracker.run_id != incoming_thread_id:
                return None
            return tracker

    def _record_event(
        self,
        tracker: _RunTracker,
        kind: EventKind,
        *,
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> ThinkEvent:
        with tracker.condition:
            tracker.sequence += 1
            event = ThinkEvent(
                sequence=tracker.sequence,
                kind=kind,
                run_id=tracker.run_id or "",
                created_at=datetime.now(timezone.utc),
                message=message,
                data=dict(data or {}),
            )
            tracker.events.append(event)
            if len(tracker.events) > _MAX_EVENT_HISTORY:
                del tracker.events[: len(tracker.events) - _MAX_EVENT_HISTORY]
            tracker.condition.notify_all()
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                logger.warning("Think progress callback failed", exc_info=True)
        return event

    @staticmethod
    def _notify_upload_progress(
        callback: UploadProgressCallback | None,
        progress: UploadProgress,
    ) -> None:
        if callback is None:
            return
        try:
            callback(progress)
        except Exception:
            logger.warning("Think upload progress callback failed", exc_info=True)

    def _handle_task_start(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        with tracker.condition:
            tracker.status = RunStatus.RUNNING
            tracker.task_ended_at = None
            tracker.refresh_requested = False
            tracker.condition.notify_all()
        self._record_event(tracker, EventKind.STARTED, data=payload)

    def _handle_task_end(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        completed = False
        final_output: str | None = None
        with tracker.condition:
            tracker.task_ended_at = self._clock()
            completed = self._mark_succeeded_locked(tracker, require_task_end=True)
            final_output = tracker.final_output
            tracker.condition.notify_all()
        if completed:
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)

    @staticmethod
    def _recompute_final_locked(tracker: _RunTracker) -> None:
        latest_final: ThinkMessage | None = None
        for message_id in tracker.message_order:
            message = tracker.messages[message_id]
            if message.type == "user_message":
                latest_final = None
            elif message.metadata.get("is_final_response") is True:
                latest_final = message
        tracker.final_message_id = latest_final.id if latest_final is not None else None
        tracker.final_output = latest_final.output if latest_final is not None else None

    @staticmethod
    def _mark_succeeded_locked(
        tracker: _RunTracker,
        *,
        require_task_end: bool,
    ) -> bool:
        final_message_id = tracker.final_message_id
        if final_message_id is None or tracker.final_output is None:
            return False
        if require_task_end and tracker.task_ended_at is None:
            return False
        tracker.status = RunStatus.SUCCEEDED
        if tracker.completed_message_id == final_message_id:
            return False
        tracker.completed_message_id = final_message_id
        return True

    def _store_message(self, tracker: _RunTracker, message: ThinkMessage) -> bool:
        with tracker.condition:
            if message.id not in tracker.messages:
                tracker.message_order.append(message.id)
            tracker.messages[message.id] = message
            self._recompute_final_locked(tracker)
            self._refresh_needs_input_prompt_locked(tracker)
            completed = self._mark_succeeded_locked(tracker, require_task_end=True)
            tracker.condition.notify_all()
            return completed

    def _refresh_needs_input_prompt_locked(self, tracker: _RunTracker) -> None:
        input_state = tracker.needs_input
        if input_state is None:
            return
        prompt = self._latest_assistant_prompt(tracker)
        if prompt is None or prompt == input_state.prompt:
            return
        tracker.needs_input = NeedsInput(
            run_id=input_state.run_id,
            kind=input_state.kind,
            prompt=prompt,
            checkpoint_id=input_state.checkpoint_id,
            details=input_state.details,
        )

    def _handle_message(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        with tracker.condition:
            run_id = tracker.run_id
        if run_id is None:
            return
        message = _message_from_payload(payload, run_id)
        if message is None:
            return
        completed = self._store_message(tracker, message)
        self._record_event(
            tracker,
            EventKind.MESSAGE,
            message=message.output,
            data={"message_id": message.id, "message_type": message.type},
        )
        if completed:
            with tracker.condition:
                final_output = tracker.final_output
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)

    def _handle_stream_token(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        message_id = _text(payload.get("id"))
        token_value = payload.get("token")
        token = token_value if isinstance(token_value, str) else None
        if message_id is None or token is None:
            return
        completed = False
        final_output: str | None = None
        with tracker.condition:
            existing = tracker.messages.get(message_id)
            if existing is None or tracker.run_id is None:
                return
            output = token if payload.get("isSequence") is True else f"{existing.output}{token}"
            tracker.messages[message_id] = ThinkMessage(
                id=existing.id,
                run_id=existing.run_id,
                type=existing.type,
                output=output,
                name=existing.name,
                metadata=existing.metadata,
            )
            self._recompute_final_locked(tracker)
            self._refresh_needs_input_prompt_locked(tracker)
            completed = self._mark_succeeded_locked(tracker, require_task_end=True)
            final_output = tracker.final_output
            tracker.condition.notify_all()
        if completed:
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)

    def _latest_assistant_prompt(self, tracker: _RunTracker) -> str | None:
        prompt: str | None = None
        for message_id in tracker.message_order:
            message = tracker.messages[message_id]
            if message.type == "user_message":
                prompt = None
                continue
            if message.type == "assistant_message" and message.output.strip():
                prompt = message.output
        return prompt

    def _handle_overlay(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        checkpoint_id = _text(payload.get("checkpointId"))
        waiting = _boolean(payload.get("waitingForInput"))
        input_state: NeedsInput | None = None
        with tracker.condition:
            stored_checkpoint = tracker.latest_checkpoint_id
            if checkpoint_id and stored_checkpoint and checkpoint_id < stored_checkpoint:
                return
            if (
                waiting
                and checkpoint_id is not None
                and tracker.answered_checkpoint_id is not None
                and checkpoint_id <= tracker.answered_checkpoint_id
            ):
                return
            if checkpoint_id:
                tracker.latest_checkpoint_id = checkpoint_id
            if waiting:
                stage = _text(payload.get("humanApprovalStage"))
                billing_details = _mapping(payload.get("billingRequired"))
                if billing_details or _text(payload.get("status")) == "mid_turn_billing_required":
                    kind = InputKind.BILLING
                    details = billing_details
                elif stage == "plan_review":
                    kind = InputKind.PLAN_REVIEW
                    details = {}
                else:
                    kind = InputKind.CLARIFICATION
                    details = {}
                input_state = NeedsInput(
                    run_id=tracker.run_id or "",
                    kind=kind,
                    prompt=self._latest_assistant_prompt(tracker),
                    checkpoint_id=checkpoint_id,
                    details=details,
                )
                tracker.needs_input = input_state
                tracker.status = RunStatus.NEEDS_INPUT
            else:
                was_waiting = (
                    tracker.needs_input is not None
                    or tracker.status is RunStatus.NEEDS_INPUT
                )
                tracker.needs_input = None
                if tracker.final_output is not None and tracker.task_ended_at is not None:
                    tracker.status = RunStatus.SUCCEEDED
                elif payload.get("visible") is True or was_waiting:
                    tracker.status = RunStatus.RUNNING
            tracker.condition.notify_all()
        display = _text(payload.get("display"))
        if input_state is not None:
            self._record_event(
                tracker,
                EventKind.NEEDS_INPUT,
                message=input_state.prompt,
                data={
                    "kind": input_state.kind.value,
                    "checkpoint_id": input_state.checkpoint_id,
                },
            )
        else:
            self._record_event(tracker, EventKind.PROGRESS, message=display, data=payload)

    def _handle_resume_thread(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        steps = payload.get("steps")
        if isinstance(steps, list):
            for raw_step in cast(list[object], steps):
                step = _mapping(raw_step)
                with tracker.condition:
                    run_id = tracker.run_id
                if run_id is None:
                    continue
                message = _message_from_payload(step, run_id)
                if message is not None:
                    self._store_message(tracker, message)
        completed = False
        final_output: str | None = None
        with tracker.condition:
            tracker.awaiting_transcript = False
            self._refresh_needs_input_prompt_locked(tracker)
            if tracker.final_output is not None:
                completed = self._mark_succeeded_locked(tracker, require_task_end=False)
            elif tracker.status in {RunStatus.SUBMITTING, RunStatus.SUCCEEDED}:
                tracker.status = RunStatus.QUEUED
            final_output = tracker.final_output
            tracker.condition.notify_all()
        if completed:
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)

    def _handle_resume_error(self, raw_error: object) -> None:
        payload = _mapping(raw_error)
        if _thread_id(payload) is not None:
            tracker = self._tracker_for_payload(payload)
        else:
            tracker = self._active_tracker_snapshot()
            if tracker is not None:
                with tracker.condition:
                    # resume() is the only writer that marks a tracker as waiting
                    # for a transcript, which safely correlates a threadless error.
                    if not tracker.awaiting_transcript:
                        tracker = None
        if tracker is None:
            return
        error_text = _text(raw_error) or _text(payload.get("error")) or "resume denied"
        failure = RunProtocolError(f"Could not resume Think run: {error_text}")
        with tracker.condition:
            tracker.failure = failure
            tracker.status = RunStatus.FAILED
            tracker.awaiting_transcript = False
            tracker.condition.notify_all()
        self._record_event(tracker, EventKind.FAILED, message=str(failure))

    def _handle_toast(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = (
            self._tracker_for_payload(payload)
            if _thread_id(payload) is not None
            else self._active_tracker_snapshot()
        )
        if tracker is None:
            return
        message = _text(payload.get("message"))
        if message:
            self._record_event(tracker, EventKind.PROGRESS, message=message, data=payload)

    def _emit_status_ready(self, run_id: str) -> None:
        self._socket.emit(
            "action",
            {
                "name": "status_client_ready",
                "id": str(uuid.uuid4()),
                "forId": None,
                "threadId": run_id,
                "payload": {},
            },
        )

    def _room_sync(self, run_id: str | None) -> None:
        event = "thread_change" if run_id else "thread_detach"
        data: object = {"threadId": run_id} if run_id else None
        try:
            raw_ack = self._socket.call(event, data, timeout=self._socket_timeout)
        except (SocketIOError, OSError) as exc:
            raise ThinkConnectionError("Could not synchronize the Think conversation") from exc
        ack = _mapping(raw_ack)
        if ack.get("success") is not True:
            raise ThinkConnectionError(
                _text(ack.get("error")) or "Think conversation synchronization failed"
            )

    def _message_metadata(
        self,
        *,
        options: RunOptions,
        files: Iterable[UploadedFile],
        datasets: Iterable[Dataset],
        conversations: Iterable[PreviousConversation] = (),
        planning_response: bool,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "mode": options.mode,
            "advancedPlanningEnabled": options.advanced_planning,
            "autoCompactEnabled": options.auto_compact,
            "fastEnabled": options.fast,
        }
        if planning_response:
            metadata["isPlanningResponse"] = True
        else:
            metadata.update(
                {
                    "verificationEnabled": options.verify,
                    "searchVerificationEnabled": options.verify_sources,
                }
            )
        if options.zero_data_retention is not None:
            metadata["zdrEnabled"] = options.zero_data_retention
        file_payloads = [
            {
                "id": uploaded.id,
                "name": uploaded.name,
                "displayName": uploaded.display_name,
                "size": uploaded.size,
                "mime": uploaded.mime,
                "scope": "personal",
            }
            for uploaded in files
        ]
        if file_payloads:
            metadata["attached_input_artifacts"] = file_payloads
        dataset_payloads: list[dict[str, object]] = []
        for dataset in datasets:
            dataset_payload: dict[str, object] = {
                "id": dataset.id,
                "name": dataset.name,
            }
            if dataset.assembly:
                dataset_payload["assembly"] = dataset.assembly
            dataset_payloads.append(dataset_payload)
        if dataset_payloads:
            metadata["attached_bystro_datasets"] = dataset_payloads
        conversation_payloads: list[dict[str, object]] = []
        for conversation in conversations:
            conversation_payload: dict[str, object] = {"id": conversation.id}
            if conversation.name:
                conversation_payload["name"] = conversation.name
            conversation_payloads.append(conversation_payload)
        if conversation_payloads:
            metadata["context_conversations"] = conversation_payloads
        return metadata

    def _client_message_payload(
        self,
        *,
        prompt: str,
        run_id: str | None,
        message_id: str,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "message": {
                "threadId": run_id or "",
                "id": message_id,
                "name": self.auth.email,
                "type": "user_message",
                "output": prompt,
                "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "metadata": dict(metadata),
            },
            "fileReferences": [],
            "new": run_id is None,
        }

    def _send_client_message(
        self,
        *,
        payload: Mapping[str, object],
        run_id: str | None,
        max_retries: int = 2,
    ) -> dict[str, object]:
        last_transport_error: Exception | None = None
        for attempt in range(max_retries + 1):
            self._room_sync(run_id)
            try:
                raw_ack = self._socket.call(
                    "client_message",
                    dict(payload),
                    timeout=self._socket_timeout,
                )
            except (SocketIOError, OSError) as exc:
                last_transport_error = exc
                if attempt < max_retries:
                    self._sleep(min(4.0, 2.0**attempt))
                    continue
                break
            except Exception:
                logger.exception("Unexpected Think message transport failure")
                raise
            ack = _mapping(raw_ack)
            if ack.get("success") is True:
                return ack
            retryable = ack.get("retryable") is True
            if retryable and attempt < max_retries:
                self._sleep(min(4.0, 2.0**attempt))
                continue
            code = _text(ack.get("error"))
            warning = _mapping(ack.get("warning"))
            message = _text(warning.get("message")) or code or "Think rejected the request"
            raise RunRejectedError(
                message,
                code=code,
                retryable=retryable,
                acknowledgement=ack,
            )
        raise ThinkConnectionError("Think did not acknowledge the request") from last_transport_error

    def _assert_can_activate(self) -> None:
        tracker = self._active_tracker_snapshot()
        if tracker is None:
            return
        with tracker.condition:
            if tracker.status not in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                raise ThinkError(
                    "This client already has an active run. Resolve or finish it, or use another client."
                )

    def _resolve_files(
        self,
        files: Iterable[FileInput],
        on_upload_progress: UploadProgressCallback | None,
    ) -> list[UploadedFile]:
        resolved: list[UploadedFile] = []
        for item in files:
            if isinstance(item, UploadedFile):
                resolved.append(item)
            else:
                resolved.append(self.upload(item, on_progress=on_upload_progress))
        return resolved

    def _resolve_context_artifacts(
        self,
        message: MessageWithContext,
    ) -> list[UploadedFile]:
        resolved: list[UploadedFile] = []
        for artifact in message.artifacts:
            if isinstance(artifact, UploadedFile):
                resolved.append(artifact)
            else:
                resolved.append(self.get_artifact(artifact.id))
        return resolved

    @staticmethod
    def _unique_files(files: Iterable[UploadedFile]) -> list[UploadedFile]:
        by_id: dict[str, UploadedFile] = {}
        for uploaded in files:
            by_id[uploaded.id] = uploaded
        return list(by_id.values())

    @staticmethod
    def _unique_datasets(datasets: Iterable[Dataset]) -> list[Dataset]:
        by_id: dict[str, Dataset] = {}
        for dataset in datasets:
            by_id[dataset.id] = dataset
        return list(by_id.values())

    def submit(
        self,
        prompt: MessageInput,
        *,
        files: Iterable[FileInput] = (),
        datasets: Iterable[Dataset] = (),
        options: RunOptions | None = None,
        on_upload_progress: UploadProgressCallback | None = None,
    ) -> "Run":
        """Upload inputs and submit a new Think workload."""

        message = prompt if isinstance(prompt, MessageWithContext) else MessageWithContext(prompt)
        normalized_prompt = message.prompt.strip()
        self._assert_can_activate()
        self.connect()
        resolved_options = options or RunOptions()
        resolved_files = self._unique_files(
            [
                *self._resolve_context_artifacts(message),
                *self._resolve_files(files, on_upload_progress),
            ]
        )
        resolved_datasets = self._unique_datasets([*message.datasets, *datasets])
        tracker = _RunTracker(run_id=None, options=resolved_options)
        with self._state_lock:
            self._active_tracker = tracker
        message_id = str(uuid.uuid4())
        metadata = self._message_metadata(
            options=resolved_options,
            files=resolved_files,
            datasets=resolved_datasets,
            conversations=message.conversations,
            planning_response=False,
        )
        payload = self._client_message_payload(
            prompt=normalized_prompt,
            run_id=None,
            message_id=message_id,
            metadata=metadata,
        )
        try:
            ack = self._send_client_message(payload=payload, run_id=None)
            run_id = _text(ack.get("threadId"))
            if run_id is None:
                raise RunProtocolError("Think accepted the workload without returning a run id")
            with tracker.condition:
                if tracker.run_id is not None and tracker.run_id != run_id:
                    raise RunProtocolError(
                        "Think acknowledged the workload for a different run"
                    )
        except Exception as exc:
            logger.debug("Think workload submission failed", exc_info=True)
            with tracker.condition:
                tracker.failure = exc
                tracker.status = RunStatus.FAILED
                tracker.condition.notify_all()
            raise
        with tracker.condition:
            tracker.run_id = run_id
            if tracker.status is RunStatus.SUBMITTING:
                tracker.status = RunStatus.QUEUED
            tracker.condition.notify_all()
        # Submission connected before its tracker existed; record admission with its run id.
        self._record_event(
            tracker,
            EventKind.CONNECTED,
            message="Connected to Think",
        )
        self._record_event(
            tracker,
            EventKind.SUBMITTED,
            message=normalized_prompt,
            data={"message_id": message_id},
        )
        self._emit_status_ready(run_id)
        return Run(self, tracker)

    def resume(self, run_id: str) -> "Run":
        """Attach to a durable Think run and hydrate its current state."""

        normalized_id = run_id.strip()
        if not normalized_id:
            raise ValueError("run_id cannot be empty")
        self._assert_can_activate()
        tracker = _RunTracker(run_id=normalized_id, options=RunOptions())
        tracker.status = RunStatus.QUEUED
        tracker.awaiting_transcript = True
        with self._state_lock:
            self._active_tracker = tracker
        was_connected = self._socket.connected
        self.connect()
        if was_connected:
            self._room_sync(normalized_id)
            self._socket.emit("connection_successful")
            self._emit_status_ready(normalized_id)
        return Run(self, tracker)

    def _synchronize_pause_checkpoint(self, tracker: _RunTracker) -> None:
        with tracker.condition:
            run_id = tracker.run_id
            input_state = tracker.needs_input
            if run_id is None:
                raise RunProtocolError("Cannot continue a run without an id")
            if tracker.status is not RunStatus.NEEDS_INPUT or input_state is None:
                raise InputResponseError("The run is not waiting for input")
            if input_state.kind is InputKind.BILLING:
                raise InputResponseError(
                    "This run is paused for billing. Resolve the billing action in the dashboard, "
                    "then call refresh()."
                )
            if input_state.checkpoint_id is not None:
                return
            last_pause = input_state

        # The live overlay is emitted before the graph checkpoint is committed.
        # A status replay is the production writer that supplies the durable id.
        # If that first replay wins the race with the checkpoint commit, the
        # server legitimately returns its preceding ``processing`` overlay.
        # Keep polling until a later replay observes the committed pause; no
        # response payload is prepared or dispatched before that happens.
        self._emit_status_ready(run_id)
        deadline = self._clock() + self._finalization_timeout
        while True:
            with tracker.condition:
                input_state = tracker.needs_input
                if input_state is not None:
                    if input_state.kind is InputKind.BILLING:
                        raise InputResponseError(
                            "This run is paused for billing. Resolve the billing action in the "
                            "dashboard, then call refresh()."
                        )
                    last_pause = input_state
                    if input_state.checkpoint_id is not None:
                        return
                if tracker.failure is not None:
                    raise tracker.failure
                if tracker.status in {
                    RunStatus.SUCCEEDED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                }:
                    raise InputResponseError("The run is no longer waiting for input")
                remaining = deadline - self._clock()
                if remaining <= 0:
                    tracker.needs_input = last_pause
                    tracker.status = RunStatus.NEEDS_INPUT
                    tracker.condition.notify_all()
                    raise RunProtocolError(
                        "Think did not return a durable needs_input checkpoint; call refresh() "
                        "and retry the response."
                    )
                tracker.condition.wait(timeout=min(_CHECKPOINT_REPLAY_INTERVAL, remaining))
            self._emit_status_ready(run_id)

    def _continue(
        self,
        tracker: _RunTracker,
        prompt: MessageInput,
        *,
        files: Iterable[FileInput],
        datasets: Iterable[Dataset],
        planning_response: bool,
        on_upload_progress: UploadProgressCallback | None,
    ) -> None:
        message = prompt if isinstance(prompt, MessageWithContext) else MessageWithContext(prompt)
        normalized_prompt = message.prompt.strip()
        with self._state_lock:
            if tracker is not self._active_tracker:
                raise InputResponseError("This run is no longer active on its client")
        if planning_response:
            self._synchronize_pause_checkpoint(tracker)
        with tracker.condition:
            run_id = tracker.run_id
            prior_status = tracker.status
            prior_input = tracker.needs_input
            prior_final_output = tracker.final_output
            prior_final_message_id = tracker.final_message_id
            prior_failure = tracker.failure
            prior_task_ended_at = tracker.task_ended_at
            prior_refresh_requested = tracker.refresh_requested
            prior_answered_checkpoint_id = tracker.answered_checkpoint_id
            if run_id is None:
                raise RunProtocolError("Cannot continue a run without an id")
            if planning_response:
                if tracker.status is not RunStatus.NEEDS_INPUT or prior_input is None:
                    raise InputResponseError("The run is not waiting for input")
                if prior_input.kind is InputKind.BILLING:
                    raise InputResponseError(
                        "This run is paused for billing. Resolve the billing action in the dashboard, "
                        "then call refresh()."
                    )
                tracker.answered_checkpoint_id = prior_input.checkpoint_id
            elif tracker.status is not RunStatus.SUCCEEDED:
                raise InputResponseError("A normal follow-up requires a completed run")
            tracker.needs_input = None
            tracker.final_output = None
            tracker.final_message_id = None
            tracker.failure = None
            tracker.task_ended_at = None
            tracker.refresh_requested = False
            tracker.status = RunStatus.SUBMITTING
            tracker.condition.notify_all()

        message_id = str(uuid.uuid4())
        try:
            resolved_files = self._unique_files(
                [
                    *self._resolve_context_artifacts(message),
                    *self._resolve_files(files, on_upload_progress),
                ]
            )
            resolved_datasets = self._unique_datasets([*message.datasets, *datasets])
            metadata = self._message_metadata(
                options=tracker.options,
                files=resolved_files,
                datasets=resolved_datasets,
                conversations=message.conversations,
                planning_response=planning_response,
            )
            payload = self._client_message_payload(
                prompt=normalized_prompt,
                run_id=run_id,
                message_id=message_id,
                metadata=metadata,
            )
            ack = self._send_client_message(payload=payload, run_id=run_id)
        except Exception:
            logger.debug("Think continuation failed before acknowledgement", exc_info=True)
            with tracker.condition:
                if tracker.status is RunStatus.SUBMITTING:
                    tracker.status = prior_status
                    tracker.needs_input = prior_input
                    tracker.final_output = prior_final_output
                    tracker.final_message_id = prior_final_message_id
                    tracker.failure = prior_failure
                    tracker.task_ended_at = prior_task_ended_at
                    tracker.refresh_requested = prior_refresh_requested
                    tracker.answered_checkpoint_id = prior_answered_checkpoint_id
                tracker.condition.notify_all()
            raise
        ack_run_id = _text(ack.get("threadId"))
        if ack_run_id != run_id:
            failure = RunProtocolError("Think acknowledged the continuation for a different run")
            with tracker.condition:
                tracker.failure = failure
                tracker.status = RunStatus.FAILED
                tracker.condition.notify_all()
            raise failure
        with tracker.condition:
            if tracker.status is RunStatus.SUBMITTING:
                tracker.status = RunStatus.QUEUED
            tracker.condition.notify_all()
        self._record_event(
            tracker,
            EventKind.SUBMITTED,
            message=normalized_prompt,
            data={"message_id": message_id, "planning_response": planning_response},
        )
        self._emit_status_ready(run_id)

    def _refresh(self, tracker: _RunTracker) -> None:
        with tracker.condition:
            run_id = tracker.run_id
        if run_id is None:
            raise RunProtocolError("Cannot refresh a run without an id")
        self.connect()
        self._room_sync(run_id)
        self._socket.emit("connection_successful")
        self._emit_status_ready(run_id)

    def upload(
        self,
        path: str | Path,
        *,
        display_name: str | None = None,
        description: str | None = None,
        artifact_path: str | None = None,
        on_progress: UploadProgressCallback | None = None,
    ) -> UploadedFile:
        """Upload a local file through Think's resumable chunk protocol."""

        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"input file does not exist: {source}")
        file_size = source.stat().st_size
        if file_size <= 0:
            raise ValueError("input file cannot be empty")
        normalized_artifact_path = _normalize_artifact_path(artifact_path, source.name)
        self.connect()
        mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        total_chunks = math.ceil(file_size / self._upload_chunk_size)
        upload_id = uuid.uuid4().hex
        sent = 0
        self._notify_upload_progress(
            on_progress,
            UploadProgress(UploadPhase.UPLOADING, sent, file_size),
        )

        with source.open("rb") as handle:
            for chunk_index in range(total_chunks):
                chunk = handle.read(self._upload_chunk_size)
                if not chunk:
                    raise RunProtocolError("input file changed while it was being uploaded")
                response_payload = self._upload_chunk(
                    chunk=chunk,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    upload_id=upload_id,
                    file_name=source.name,
                    file_size=file_size,
                    mime_type=mime_type,
                    display_name=display_name,
                    description=description,
                    artifact_path=normalized_artifact_path,
                )
                sent += len(chunk)
                self._notify_upload_progress(
                    on_progress,
                    UploadProgress(UploadPhase.UPLOADING, sent, file_size),
                )
                if response_payload.get("completed") is True:
                    file_payload = _mapping(response_payload.get("file"))
                    uploaded = _uploaded_file(file_payload)
                    self._notify_upload_progress(
                        on_progress,
                        UploadProgress(UploadPhase.COMPLETE, file_size, file_size),
                    )
                    return uploaded

        self._notify_upload_progress(
            on_progress,
            UploadProgress(UploadPhase.FINALIZING, file_size, file_size),
        )
        uploaded = self._poll_upload(upload_id)
        self._notify_upload_progress(
            on_progress,
            UploadProgress(UploadPhase.COMPLETE, file_size, file_size),
        )
        return uploaded

    def upload_artifact(
        self,
        path: str | Path,
        *,
        display_name: str | None = None,
        description: str | None = None,
        artifact_path: str | None = None,
        on_progress: UploadProgressCallback | None = None,
    ) -> UploadedFile:
        """Upload a file to personal artifacts (explicitly named alias)."""

        return self.upload(
            path,
            display_name=display_name,
            description=description,
            artifact_path=artifact_path,
            on_progress=on_progress,
        )

    def get_artifact(self, artifact_id: str) -> UploadedFile:
        """Resolve an existing personal artifact by id."""

        normalized_id = artifact_id.strip()
        if not normalized_id:
            raise ValueError("artifact_id cannot be empty")
        self.connect()
        try:
            response = self._session.get(
                f"{self.think_url}/user/files/{quote(normalized_id, safe='')}",
                timeout=self._http_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ThinkConnectionError("Could not fetch the Think artifact") from exc
        payload = _raise_for_response(response, action="artifact lookup")
        return _uploaded_file(payload)

    def _upload_chunk(
        self,
        *,
        chunk: bytes,
        chunk_index: int,
        total_chunks: int,
        upload_id: str,
        file_name: str,
        file_size: int,
        mime_type: str,
        display_name: str | None,
        description: str | None,
        artifact_path: str | None,
    ) -> dict[str, object]:
        data: dict[str, str] = {
            "chunkIndex": str(chunk_index),
            "totalChunks": str(total_chunks),
            "uploadId": upload_id,
            "fileName": file_name,
            "fileSize": str(file_size),
            "mimeType": mime_type,
            "chunkChecksum": hashlib.sha256(chunk).hexdigest(),
        }
        if display_name:
            data["display_name"] = display_name
        if description:
            data["description"] = description
        if artifact_path:
            data["artifact_path"] = artifact_path
        for attempt in range(self._upload_max_retries + 1):
            try:
                response = self._session.post(
                    f"{self.think_url}/user/files/chunk",
                    data=data,
                    files={"chunk": (file_name, chunk, mime_type)},
                    timeout=self._http_timeout,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt < self._upload_max_retries:
                    self._sleep(min(15.0, 2.0**attempt))
                    continue
                raise ThinkConnectionError("File upload failed after retries") from exc
            except Exception:
                logger.exception("Unexpected Think chunk upload failure")
                raise
            if 200 <= response.status_code < 300:
                return _json_payload(response)
            retryable = response.status_code in {408, 429} or response.status_code >= 500
            if retryable and attempt < self._upload_max_retries:
                self._sleep(min(15.0, 2.0**attempt))
                continue
            _raise_for_response(response, action="file upload")
        raise ThinkConnectionError("File upload failed after retries")

    def _poll_upload(self, upload_id: str) -> UploadedFile:
        started = self._clock()
        attempt = 0
        query = urlencode({"uploadId": upload_id})
        while self._clock() - started < self._upload_finalize_timeout:
            response: _Response | None = None
            try:
                response = self._session.get(
                    f"{self.think_url}/user/files/chunk/status?{query}",
                    timeout=self._http_timeout,
                    allow_redirects=False,
                )
            except requests.RequestException:
                logger.debug("Think upload finalization poll failed", exc_info=True)
            except Exception:
                logger.exception("Unexpected Think upload polling failure")
                raise
            if response is not None and 200 <= response.status_code < 300:
                payload = _json_payload(response)
                status = _text(payload.get("status"))
                if status == "done":
                    return _uploaded_file(_mapping(payload.get("file")))
                if status == "error":
                    raise RunProtocolError(
                        _text(payload.get("error")) or "Think upload finalization failed"
                    )
            elif response is not None:
                retryable = (
                    response.status_code in {408, 429}
                    or response.status_code >= 500
                )
                if not retryable:
                    _raise_for_response(response, action="upload finalization")
            delay = min(15.0, 2.0 ** min(attempt, 4))
            attempt += 1
            self._sleep(delay)
        raise RunTimeoutError("Think upload finalization timed out")

    def close(self) -> None:
        """Close live and HTTP transports without cancelling durable work."""

        with self._connect_lock:
            if self._closed:
                return
            self._closed = True
            if self._socket.connected:
                try:
                    self._socket.disconnect()
                except (SocketIOError, OSError):
                    logger.debug("Think socket disconnect failed", exc_info=True)
                except Exception:
                    logger.warning("Unexpected Think socket disconnect failure", exc_info=True)
            if self._owns_session:
                self._session.close()


class Run:
    """Handle to a durable Think conversation and its current turn."""

    def __init__(self, client: ThinkClient, tracker: _RunTracker) -> None:
        self._client = client
        self._tracker = tracker

    @property
    def id(self) -> str:
        with self._tracker.condition:
            if self._tracker.run_id is None:
                raise RunProtocolError("Think run id is not available yet")
            return self._tracker.run_id

    @property
    def status(self) -> RunStatus:
        with self._tracker.condition:
            return self._tracker.status

    @property
    def needs_input(self) -> NeedsInput | None:
        with self._tracker.condition:
            return self._tracker.needs_input

    @property
    def messages(self) -> tuple[ThinkMessage, ...]:
        with self._tracker.condition:
            return tuple(
                self._tracker.messages[message_id]
                for message_id in self._tracker.message_order
            )

    @property
    def history(self) -> tuple[ThinkEvent, ...]:
        with self._tracker.condition:
            return tuple(self._tracker.events)

    def _outcome_locked(self) -> RunOutcome | None:
        if self._tracker.status is RunStatus.NEEDS_INPUT:
            if self._tracker.needs_input is None:
                raise RunProtocolError("Run is marked needs_input without input details")
            if self._tracker.awaiting_transcript:
                return None
            return self._tracker.needs_input
        if self._tracker.status is RunStatus.SUCCEEDED:
            if self._tracker.final_output is None:
                raise RunProtocolError("Run succeeded without a final response")
            return RunResult(run_id=self.id, output=self._tracker.final_output)
        return None

    def _terminal_state_locked(
        self,
    ) -> tuple[RunOutcome | None, Exception | None, bool]:
        outcome = self._outcome_locked()
        failure = self._tracker.failure
        should_refresh = False
        if (
            failure is None
            and outcome is None
            and self._tracker.task_ended_at is not None
        ):
            elapsed = self._client._clock() - self._tracker.task_ended_at
            if elapsed >= 1.0 and not self._tracker.refresh_requested:
                self._tracker.refresh_requested = True
                should_refresh = True
            elif elapsed >= self._client._finalization_timeout:
                failure = RunProtocolError(
                    "Think ended the task without a final response or needs_input state"
                )
                self._tracker.failure = failure
                self._tracker.status = RunStatus.FAILED
                self._tracker.condition.notify_all()
        return outcome, failure, should_refresh

    def _events_after_locked(self, sequence: int) -> tuple[ThinkEvent, ...]:
        events = self._tracker.events
        if not events:
            return ()
        first_sequence = events[0].sequence
        start = max(0, sequence - first_sequence + 1)
        return tuple(events[start:])

    def wait(
        self,
        timeout: float | None = None,
        *,
        on_event: ProgressCallback | None = None,
    ) -> RunOutcome:
        """Block until the turn succeeds, pauses for input, or fails."""

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        deadline = None if timeout is None else self._client._clock() + timeout
        cursor = 0
        while True:
            callbacks: tuple[ThinkEvent, ...]
            outcome: RunOutcome | None
            failure: Exception | None
            should_refresh = False
            with self._tracker.condition:
                callbacks = self._events_after_locked(cursor)
                if callbacks:
                    cursor = callbacks[-1].sequence
                outcome, failure, should_refresh = self._terminal_state_locked()
                now = self._client._clock()
                remaining = None if deadline is None else deadline - now
            if on_event is not None:
                for event in callbacks:
                    try:
                        on_event(event)
                    except Exception:
                        logger.warning("Think wait progress callback failed", exc_info=True)
            if failure is not None:
                raise failure
            if outcome is not None:
                return outcome
            if callbacks:
                # Progress callbacks may synchronously advance this run. Re-sample
                # tracker state before waiting so their notification is not lost.
                continue
            if should_refresh:
                self.refresh()
                continue
            if remaining is not None and remaining <= 0:
                raise RunTimeoutError(
                    f"Timed out waiting for Think run {self.id} (status={self.status.value})"
                )
            wait_for = remaining
            if self._tracker.task_ended_at is not None:
                wait_for = 0.25 if remaining is None else min(0.25, remaining)
            with self._tracker.condition:
                self._tracker.condition.wait(timeout=wait_for)

    def events(self, timeout: float | None = None) -> Iterator[ThinkEvent]:
        """Yield ordered events until the current turn blocks or completes."""

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        deadline = None if timeout is None else self._client._clock() + timeout
        cursor = 0
        pending_events: deque[ThinkEvent] = deque()
        while True:
            event: ThinkEvent | None = None
            outcome: RunOutcome | None = None
            failure: Exception | None = None
            should_refresh = False
            with self._tracker.condition:
                if not pending_events:
                    pending_events.extend(self._events_after_locked(cursor))
                if pending_events:
                    event = pending_events.popleft()
                    cursor = event.sequence
                else:
                    outcome, failure, should_refresh = self._terminal_state_locked()
                    remaining = (
                        None if deadline is None else deadline - self._client._clock()
                    )
                    if failure is None and outcome is None and not should_refresh:
                        if remaining is not None and remaining <= 0:
                            failure = RunTimeoutError(
                                f"Timed out streaming Think run {self.id}"
                            )
                        else:
                            wait_for = remaining
                            if self._tracker.task_ended_at is not None:
                                wait_for = (
                                    0.25
                                    if remaining is None
                                    else min(0.25, remaining)
                                )
                            self._tracker.condition.wait(timeout=wait_for)
                            continue
            if failure is not None:
                raise failure
            if event is not None:
                yield event
                continue
            if outcome is not None:
                return
            if should_refresh:
                self.refresh()

    def respond(
        self,
        response: MessageInput,
        *,
        files: Iterable[FileInput] = (),
        datasets: Iterable[Dataset] = (),
        on_upload_progress: UploadProgressCallback | None = None,
    ) -> Self:
        """Answer a clarification or approve/revise a proposed plan."""

        self._client._continue(
            self._tracker,
            response,
            files=files,
            datasets=datasets,
            planning_response=True,
            on_upload_progress=on_upload_progress,
        )
        return self

    def follow_up(
        self,
        prompt: MessageInput,
        *,
        files: Iterable[FileInput] = (),
        datasets: Iterable[Dataset] = (),
        on_upload_progress: UploadProgressCallback | None = None,
    ) -> Self:
        """Start another turn in a successfully completed conversation."""

        self._client._continue(
            self._tracker,
            prompt,
            files=files,
            datasets=datasets,
            planning_response=False,
            on_upload_progress=on_upload_progress,
        )
        return self

    def refresh(self) -> Self:
        """Rehydrate durable transcript and pause state after reconnecting."""

        self._client._refresh(self._tracker)
        return self


__all__ = [
    "DEFAULT_THINK_URL",
    "ProgressCallback",
    "Run",
    "ThinkClient",
    "UploadProgressCallback",
]
