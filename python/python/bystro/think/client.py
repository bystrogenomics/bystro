"""Synchronous, event-driven client for Bystro Think workloads."""

# ``Run`` is intentionally a friend handle over ``ThinkClient`` internals.
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
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
    RunCancelledError,
    RunFailedError,
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
    BillingTopUpApproval,
    BillingTopUpRequest,
    ConversationMode,
    Dataset,
    EventKind,
    FileInput,
    InputKind,
    NeedsInput,
    OutputFile,
    ProgressPhase,
    ProgressUpdate,
    RunOptions,
    RunOutcome,
    RunResult,
    RunStatus,
    StreamOperation,
    StreamUpdate,
    ThinkEvent,
    ThinkMessage,
    UploadedFile,
    UploadPhase,
    UploadProgress,
)
from bystro.think.progress import (
    ProgressCallback,
    ProgressRenderer,
    close_progress_callback,
    show_progress as show_progress,
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
_ALLOWED_TRANSPORTS = frozenset({"polling", "websocket"})

UploadProgressCallback: TypeAlias = Callable[[UploadProgress], None]
FileInputs: TypeAlias = FileInput | Iterable[FileInput]
TransportInputs: TypeAlias = str | Iterable[str]
InputCallback: TypeAlias = Callable[[NeedsInput], MessageInput]
BillingApprovalCallback: TypeAlias = Callable[
    [BillingTopUpRequest], BillingTopUpApproval | None
]


class _Response(Protocol):
    status_code: int
    reason: str

    def json(self) -> object: ...

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class _CookieJar(Protocol):
    def set(  # noqa: A003 - mirrors the requests cookie-jar interface
        self, name: str, value: str, **kwargs: object
    ) -> object: ...


class _HTTPSession(Protocol):
    @property
    def cookies(self) -> _CookieJar: ...

    def post(self, url: str, **kwargs: object) -> _Response: ...

    def put(self, url: str, **kwargs: object) -> _Response: ...

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


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    try:
        normalized = float(value)
    except OverflowError:
        return None
    return normalized if math.isfinite(normalized) else None


def _message_id(idempotency_key: str | None, *, run_id: str | None) -> str:
    if idempotency_key is None:
        return str(uuid.uuid4())
    if not idempotency_key.strip():
        raise ValueError("idempotency_key cannot be empty")
    scope = run_id or "new"
    seed = (
        f"https://api.bystro.com/think/messages/{scope}/{idempotency_key}".encode()
    )
    digest = hashlib.sha256(seed).digest()
    # Chainlit accepts only UUIDv4 client-message IDs. Passing version=4 sets
    # the RFC version and variant bits while retaining 122 deterministic bits.
    return str(uuid.UUID(bytes=digest[:16], version=4))


def _optional_nonnegative_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    normalized = _nonnegative_float(value)
    if normalized is None:
        raise RunProtocolError(f"Think returned an invalid {field_name}")
    return normalized


def _normalize_base_url(url: str) -> str:
    normalized = dashboard_auth.normalize_url(url)
    parsed = urlsplit(normalized)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _normalize_transports(transports: TransportInputs | None) -> tuple[str, ...] | None:
    if transports is None:
        return None
    candidates = (transports,) if isinstance(transports, str) else tuple(transports)
    if not candidates:
        raise ValueError("Think transports cannot be empty")
    if any(
        not isinstance(candidate, str) or candidate not in _ALLOWED_TRANSPORTS
        for candidate in candidates
    ):
        allowed = ", ".join(sorted(_ALLOWED_TRANSPORTS))
        raise ValueError(f"unsupported Think transport; expected one of: {allowed}")
    return tuple(dict.fromkeys(candidates))


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


def _normalize_output_path(output_path: str) -> str:
    normalized = output_path.strip()
    if not normalized or normalized.startswith(("/", "\\")) or "\\" in normalized:
        raise ValueError("output file path must be a non-empty relative path")
    parts = normalized.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError("output file path contains invalid components")
    return normalized


def _optional_utc_datetime(value: object, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    raw_value = _text(value)
    if raw_value is None:
        raise RunProtocolError(f"Think returned an invalid {field_name}")
    normalized = raw_value[:-1] + "+00:00" if raw_value.endswith("Z") else raw_value
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise RunProtocolError(f"Think returned an invalid {field_name}") from exc


def _result_execution_metadata(
    metadata: Mapping[str, object],
) -> tuple[datetime | None, datetime | None, float | None]:
    return (
        _optional_utc_datetime(
            metadata.get("agent_execution_started_at"),
            field_name="agent execution start time",
        ),
        _optional_utc_datetime(
            metadata.get("agent_execution_completed_at"),
            field_name="agent execution completion time",
        ),
        _optional_nonnegative_float(
            metadata.get("agent_execution_duration_seconds"),
            field_name="agent execution duration",
        ),
    )


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


def _is_billing_rejection(error: RunRejectedError) -> bool:
    billing = _mapping(error.acknowledgement.get("billing"))
    return (
        error.code == "INSUFFICIENT_BILLING_CREDITS"
        or _text(billing.get("status")) == "billing_required"
    )


def _billing_top_up_request(error: RunRejectedError) -> BillingTopUpRequest | None:
    if not _is_billing_rejection(error):
        return None
    billing = _mapping(error.acknowledgement.get("billing"))
    if _text(billing.get("reservationStatus")) == "pending":
        return None
    if _text(billing.get("spendCapMode")) == "unlimited":
        return None
    required_cost = _nonnegative_int(billing.get("requiredCostCents"))
    additional_cost = _nonnegative_int(billing.get("additionalCostCents"))
    minimum_top_up = additional_cost if additional_cost else required_cost
    if minimum_top_up is None or minimum_top_up <= 0:
        return None
    return BillingTopUpRequest(
        minimum_top_up_cents=minimum_top_up,
        required_cost_cents=required_cost or minimum_top_up,
        current_monthly_limit_cents=_nonnegative_int(
            billing.get("spendCapMonthlyLimitCents")
        ),
        message=_text(billing.get("accessMessage")) or str(error),
        details=billing,
    )


def _billing_required_error(error: RunRejectedError) -> ThinkBillingRequiredError:
    billing = _mapping(error.acknowledgement.get("billing"))
    pending = _text(billing.get("reservationStatus")) == "pending"
    return ThinkBillingRequiredError(
        _text(billing.get("accessMessage")) or str(error),
        status_code=402,
        code=error.code or "billing_required",
        retryable=pending,
        request=_billing_top_up_request(error),
        action_url=_text(billing.get("checkoutUrl")) or _text(billing.get("portalUrl")),
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
        is_error=payload.get("isError") is True,
    )


def _stream_traits(message: ThinkMessage) -> tuple[str | None, bool]:
    stream_type = _text(message.metadata.get("stream_type"))
    is_reasoning = message.metadata.get("is_reasoning") is True or stream_type in {
        "reasoning",
        "thinking",
    }
    return stream_type, is_reasoning


def _progress_update(message: ThinkMessage) -> ProgressUpdate | None:
    """Parse the backend progress card without trusting its metadata shape."""

    if _text(message.metadata.get("section")) != "progress":
        return None
    raw_progress = _mapping(message.metadata.get("progress"))
    raw_phases = raw_progress.get("phases")
    if not isinstance(raw_phases, list):
        return ProgressUpdate(done=_boolean(raw_progress.get("done")), phases=())
    phases: list[ProgressPhase] = []
    for raw_phase in cast(list[object], raw_phases):
        phase = _mapping(raw_phase)
        phase_id = _text(phase.get("id"))
        kind = _text(phase.get("kind"))
        state = _text(phase.get("state"))
        label = _text(phase.get("label"))
        if None in {phase_id, kind, state, label}:
            continue
        count = _mapping(phase.get("count"))
        started_at: datetime | None = None
        started_at_ms = _nonnegative_float(phase.get("started_at"))
        if started_at_ms is not None:
            try:
                started_at = datetime.fromtimestamp(
                    started_at_ms / 1000,
                    timezone.utc,
                )
            except (OverflowError, OSError, ValueError):
                started_at = None
        phases.append(
            ProgressPhase(
                id=cast(str, phase_id),
                kind=cast(str, kind),
                state=cast(str, state),
                label=cast(str, label),
                detail=_text(phase.get("detail")),
                completed=_nonnegative_int(count.get("done")),
                total=_nonnegative_int(count.get("total")),
                started_at=started_at,
                duration_seconds=_nonnegative_float(phase.get("duration_s")),
            )
        )
    return ProgressUpdate(
        done=_boolean(raw_progress.get("done")),
        phases=tuple(phases),
    )


_AGENT_ACTIVITY_LABELS: frozenset[str] = frozenset(
    ("Thinking", "Using tools", "Generating text", "Sketching the Python analysis")
)


def _agent_activity(message: ThinkMessage) -> str | None:
    """Read the live activity label from the agent-execution step."""

    if (
        message.type != "run"
        or _text(message.metadata.get("section")) != "intermediate_outputs"
    ):
        return None
    activity = _text(message.metadata.get("activity"))
    return activity if activity in _AGENT_ACTIVITY_LABELS else None


def _run_options_from_metadata(
    metadata: Mapping[str, object],
    current: RunOptions,
) -> RunOptions:
    raw_mode = _text(metadata.get("mode"))
    mode = current.mode
    if raw_mode in {"base", "plus", "plus2", "phd"}:
        mode = cast(ConversationMode, raw_mode)

    def boolean(key: str, default: bool) -> bool:
        value = metadata.get(key)
        return value if isinstance(value, bool) else default

    raw_zdr = metadata.get("zdrEnabled")
    zero_data_retention = (
        raw_zdr if isinstance(raw_zdr, bool) else current.zero_data_retention
    )
    return RunOptions(
        mode=mode,
        advanced_planning=boolean(
            "advancedPlanningEnabled",
            current.advanced_planning,
        ),
        auto_compact=boolean("autoCompactEnabled", current.auto_compact),
        fast=boolean("fastEnabled", current.fast),
        verify=boolean("verificationEnabled", current.verify),
        verify_sources=boolean(
            "searchVerificationEnabled",
            current.verify_sources,
        ),
        zero_data_retention=zero_data_retention,
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


def _output_file(payload: Mapping[str, object]) -> OutputFile:
    name = _text(payload.get("name"))
    raw_path = _text(payload.get("path"))
    raw_size = payload.get("size")
    raw_modified = payload.get("modified")
    raw_created = payload.get("created")
    if name is None or raw_path is None:
        raise RunProtocolError("Think output listing omitted a file name or path")
    path = _normalize_output_path(raw_path)
    if path.rsplit("/", maxsplit=1)[-1] != name:
        raise RunProtocolError("Think output listing returned a mismatched file name")
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise RunProtocolError("Think output listing returned an invalid file size")
    if (
        isinstance(raw_modified, bool)
        or not isinstance(raw_modified, (int, float))
        or isinstance(raw_created, bool)
        or not isinstance(raw_created, (int, float))
    ):
        raise RunProtocolError("Think output listing returned invalid timestamps")
    return OutputFile(name, path, raw_size, float(raw_modified), float(raw_created))


@dataclass(slots=True)
class _RunTracker:
    run_id: str | None
    options: RunOptions
    condition: threading.Condition = field(default_factory=threading.Condition)
    status: RunStatus = RunStatus.SUBMITTING
    events: list[ThinkEvent] = field(default_factory=list)
    messages: dict[str, ThinkMessage] = field(default_factory=dict)
    message_order: list[str] = field(default_factory=list)
    streamed_message_ids: set[str] = field(default_factory=set)
    needs_input: NeedsInput | None = None
    final_output: str | None = None
    failure: Exception | None = None
    sequence: int = 0
    latest_checkpoint_id: str | None = None
    answered_checkpoint_id: str | None = None
    final_message_id: str | None = None
    completed_message_id: str | None = None
    final_result: RunResult | None = None
    task_ended_at: float | None = None
    task_id: str | None = None
    cancel_requested: bool = False
    cancelling_event_emitted: bool = False
    stop_released: bool = False
    refresh_requested: bool = False
    awaiting_transcript: bool = False
    progress_callback: ProgressCallback | None = None


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
        on_billing_required: BillingApprovalCallback | None = None,
        upload_chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
        upload_max_retries: int = 8,
        upload_finalize_timeout: float = 24 * 60 * 60,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        http_timeout: tuple[float, float] = _DEFAULT_HTTP_TIMEOUT,
        finalization_timeout: float = _DEFAULT_FINALIZATION_TIMEOUT,
        transports: TransportInputs | None = None,
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
        self._on_billing_required = on_billing_required
        self._upload_chunk_size = upload_chunk_size
        self._upload_max_retries = upload_max_retries
        self._upload_finalize_timeout = upload_finalize_timeout
        self._socket_timeout = socket_timeout
        self._http_timeout = http_timeout
        self._finalization_timeout = finalization_timeout
        self._transports = _normalize_transports(transports)
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
        on_billing_required: BillingApprovalCallback | None = None,
        upload_chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
        upload_max_retries: int = 8,
        upload_finalize_timeout: float = 24 * 60 * 60,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        http_timeout: tuple[float, float] = _DEFAULT_HTTP_TIMEOUT,
        finalization_timeout: float = _DEFAULT_FINALIZATION_TIMEOUT,
        transports: TransportInputs | None = None,
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
            on_billing_required=on_billing_required,
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
        on_billing_required: BillingApprovalCallback | None = None,
        upload_chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
        upload_max_retries: int = 8,
        upload_finalize_timeout: float = 24 * 60 * 60,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        http_timeout: tuple[float, float] = _DEFAULT_HTTP_TIMEOUT,
        finalization_timeout: float = _DEFAULT_FINALIZATION_TIMEOUT,
        transports: TransportInputs | None = None,
    ) -> Self:
        """Construct a client from ``~/.bystro`` authentication state."""

        return cls(
            think_url=think_url,
            on_event=on_event,
            on_billing_required=on_billing_required,
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
            "task_stopping": self._handle_task_stopping,
            "thread_stop_released": self._handle_thread_stop_released,
            "new_message": self._handle_message,
            "update_message": self._handle_update_message,
            "delete_message": self._handle_delete_message,
            "stream_start": self._handle_stream_start,
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
            with tracker.condition:
                cancel_requested = tracker.cancel_requested and not tracker.stop_released
            if cancel_requested:
                self._emit_stop_request(tracker)

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
        stream_update: StreamUpdate | None = None,
        progress: ProgressUpdate | None = None,
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
                stream_update=stream_update,
                progress=progress,
            )
            tracker.events.append(event)
            if len(tracker.events) > _MAX_EVENT_HISTORY:
                del tracker.events[: len(tracker.events) - _MAX_EVENT_HISTORY]
            tracker.condition.notify_all()
        callback = tracker.progress_callback or self._on_event
        if callback is not None:
            try:
                callback(event)
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
            if tracker.status in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return
            tracker.task_id = _text(payload.get("taskId")) or tracker.task_id
            tracker.task_ended_at = None
            tracker.refresh_requested = False
            cancel_requested = tracker.cancel_requested
            tracker.status = (
                RunStatus.CANCELLING
                if cancel_requested
                else RunStatus.RUNNING
            )
            tracker.condition.notify_all()
        self._record_event(tracker, EventKind.STARTED, data=payload)
        if cancel_requested:
            self._emit_stop_request(tracker)

    def _handle_task_end(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        completed = False
        final_output: str | None = None
        with tracker.condition:
            if tracker.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
                return
            incoming_task_id = _text(payload.get("taskId"))
            if (
                incoming_task_id is not None
                and tracker.task_id is not None
                and incoming_task_id != tracker.task_id
            ):
                return
            tracker.task_ended_at = self._clock()
            tracker.task_id = incoming_task_id or tracker.task_id
            if tracker.cancel_requested:
                tracker.status = RunStatus.CANCELLING
            else:
                completed = self._mark_succeeded_locked(
                    tracker,
                    require_task_end=True,
                )
            final_output = tracker.final_output
            tracker.condition.notify_all()
        if completed:
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)

    def _handle_task_stopping(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        with tracker.condition:
            if tracker.stop_released or tracker.status not in {
                RunStatus.SUBMITTING,
                RunStatus.QUEUED,
                RunStatus.RUNNING,
                RunStatus.CANCELLING,
            }:
                return
            incoming_task_id = _text(payload.get("taskId"))
            if (
                incoming_task_id is not None
                and tracker.task_id is not None
                and incoming_task_id != tracker.task_id
            ):
                return
            tracker.cancel_requested = True
            tracker.task_id = incoming_task_id or tracker.task_id
            tracker.status = RunStatus.CANCELLING
            should_emit = not tracker.cancelling_event_emitted
            tracker.cancelling_event_emitted = True
            tracker.condition.notify_all()
        if should_emit:
            self._record_event(
                tracker,
                EventKind.CANCELLING,
                message="Cancelling analysis",
                data=payload,
            )

    def _handle_thread_stop_released(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        with tracker.condition:
            incoming_task_id = _text(payload.get("taskId"))
            if tracker.stop_released or not tracker.cancel_requested or (
                incoming_task_id is not None
                and tracker.task_id is not None
                and incoming_task_id != tracker.task_id
            ):
                return
            tracker.stop_released = True
            tracker.task_id = incoming_task_id or tracker.task_id
            tracker.status = RunStatus.CANCELLED
            tracker.failure = RunCancelledError(
                f"Think run {tracker.run_id or ''} was cancelled"
            )
            tracker.needs_input = None
        self._record_event(
            tracker,
            EventKind.CANCELLED,
            message="Analysis cancelled",
            data=payload,
        )

    @staticmethod
    def _recompute_final_locked(tracker: _RunTracker) -> None:
        previous_final_message_id = tracker.final_message_id
        previous_final_output = tracker.final_output
        latest_final: ThinkMessage | None = None
        for message_id in tracker.message_order:
            message = tracker.messages[message_id]
            if message.type == "user_message":
                latest_final = None
            elif message.metadata.get("is_final_response") is True:
                latest_final = message
        tracker.final_message_id = latest_final.id if latest_final is not None else None
        tracker.final_output = latest_final.output if latest_final is not None else None
        if (
            tracker.final_message_id != previous_final_message_id
            or tracker.final_output != previous_final_output
        ):
            tracker.final_result = None

    @staticmethod
    def _latest_turn_was_manually_stopped_locked(tracker: _RunTracker) -> bool:
        """Return whether the latest durable turn ended at a Stop marker."""

        stopped = False
        for message_id in tracker.message_order:
            message = tracker.messages[message_id]
            if (
                message.type == "user_message"
                or message.metadata.get("is_final_response") is True
            ):
                stopped = False
            elif message.metadata.get("manual_stop_status") is True:
                stopped = True
        return stopped

    @staticmethod
    def _mark_succeeded_locked(
        tracker: _RunTracker,
        *,
        require_task_end: bool,
    ) -> bool:
        if tracker.cancel_requested or tracker.failure is not None:
            return False
        final_message_id = tracker.final_message_id
        if final_message_id is None or tracker.final_output is None:
            return False
        if require_task_end and tracker.task_ended_at is None:
            return False
        try:
            _result_execution_metadata(tracker.messages[final_message_id].metadata)
        except RunProtocolError as exc:
            tracker.failure = exc
            tracker.status = RunStatus.FAILED
            tracker.needs_input = None
            return False
        tracker.status = RunStatus.SUCCEEDED
        if tracker.completed_message_id == final_message_id:
            return False
        tracker.completed_message_id = final_message_id
        return True

    @staticmethod
    def _latest_turn_error_locked(tracker: _RunTracker) -> ThinkMessage | None:
        latest_error: ThinkMessage | None = None
        for message_id in tracker.message_order:
            message = tracker.messages[message_id]
            if message.type == "user_message":
                latest_error = None
            elif message.is_error:
                latest_error = message
            elif message.metadata.get("is_final_response") is True:
                latest_error = None
        return latest_error

    def _store_message(self, tracker: _RunTracker, message: ThinkMessage) -> bool:
        with tracker.condition:
            if message.id not in tracker.messages:
                tracker.message_order.append(message.id)
            tracker.messages[message.id] = message
            self._recompute_final_locked(tracker)
            self._refresh_needs_input_prompt_locked(tracker)
            completed = self._mark_succeeded_locked(
                tracker,
                require_task_end=True,
            )
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

    @staticmethod
    def _stream_update(
        message: ThinkMessage,
        previous_output: str | None,
    ) -> StreamUpdate | None:
        """Convert cumulative Socket.IO snapshots into bounded incremental updates."""

        if message.type != "assistant_message":
            return None
        stream_type, is_reasoning = _stream_traits(message)
        section = _text(message.metadata.get("section"))
        if is_reasoning:
            if previous_output is not None:
                return None
            return StreamUpdate(
                message_id=message.id,
                delta="",
                operation="append",
                content_length=len(message.output),
                message_type=message.type,
                name=message.name,
                stream_type=stream_type,
                is_reasoning=True,
                section=section,
            )
        if previous_output is None or message.output.startswith(previous_output):
            delta = message.output[len(previous_output or "") :]
            operation: StreamOperation = "append"
        else:
            delta = message.output
            operation = "replace"
        if not delta:
            return None
        return StreamUpdate(
            message_id=message.id,
            delta=delta,
            operation=operation,
            content_length=len(message.output),
            message_type=message.type,
            name=message.name,
            stream_type=stream_type,
            is_reasoning=False,
            section=section,
        )

    def _record_stream_update(
        self,
        tracker: _RunTracker,
        message: ThinkMessage,
        *,
        previous_output: str | None,
    ) -> None:
        with tracker.condition:
            tracker.streamed_message_ids.add(message.id)
        update = self._stream_update(message, previous_output)
        if update is not None:
            self._record_event(
                tracker,
                EventKind.STREAM,
                stream_update=update,
            )

    def _handle_message_payload(
        self,
        raw_payload: object,
        *,
        stream_start: bool,
        continue_stream: bool,
    ) -> None:
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
        with tracker.condition:
            previous = tracker.messages.get(message.id)
            was_streamed = message.id in tracker.streamed_message_ids
        completed = self._store_message(tracker, message)
        stream_type, is_reasoning = _stream_traits(message)
        progress = _progress_update(message)
        activity = _agent_activity(message)
        event_data: dict[str, object] = {
            "message_id": message.id,
            "message_type": message.type,
            "is_reasoning": is_reasoning,
        }
        if stream_type is not None:
            event_data["stream_type"] = stream_type
        section = _text(message.metadata.get("section"))
        if section is not None:
            event_data["section"] = section
        if activity is not None:
            event_data["activity"] = activity
        if message.is_error:
            event_data["is_error"] = True
        if progress is not None:
            self._record_event(
                tracker,
                EventKind.PROGRESS,
                message=message.output,
                data=event_data,
                progress=progress,
            )
        elif activity is not None:
            self._record_event(
                tracker,
                EventKind.PROGRESS,
                message=activity,
                data=event_data,
            )
        else:
            self._record_event(
                tracker,
                EventKind.MESSAGE,
                message=None if is_reasoning else message.output,
                data=event_data,
            )
        if progress is None and (stream_start or (continue_stream and was_streamed)):
            previous_output = (
                previous.output if was_streamed and previous is not None else None
            )
            self._record_stream_update(
                tracker,
                message,
                previous_output=previous_output,
            )
        if completed:
            with tracker.condition:
                final_output = tracker.final_output
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)

    def _handle_message(self, raw_payload: object) -> None:
        self._handle_message_payload(
            raw_payload,
            stream_start=False,
            continue_stream=False,
        )

    def _handle_update_message(self, raw_payload: object) -> None:
        self._handle_message_payload(
            raw_payload,
            stream_start=False,
            continue_stream=True,
        )

    def _handle_stream_start(self, raw_payload: object) -> None:
        self._handle_message_payload(
            raw_payload,
            stream_start=True,
            continue_stream=False,
        )

    def _handle_delete_message(self, raw_payload: object) -> None:
        payload = _mapping(raw_payload)
        tracker = self._tracker_for_payload(payload)
        if tracker is None:
            return
        message_id = _text(payload.get("id"))
        if message_id is None:
            return
        with tracker.condition:
            removed = tracker.messages.pop(message_id, None)
            if message_id in tracker.message_order:
                tracker.message_order.remove(message_id)
            was_streamed = message_id in tracker.streamed_message_ids
            tracker.streamed_message_ids.discard(message_id)
            self._recompute_final_locked(tracker)
            self._refresh_needs_input_prompt_locked(tracker)
            tracker.condition.notify_all()
        if was_streamed and removed is not None:
            stream_type, is_reasoning = _stream_traits(removed)
            self._record_event(
                tracker,
                EventKind.STREAM,
                stream_update=StreamUpdate(
                    message_id=removed.id,
                    delta="",
                    operation="retract",
                    content_length=0,
                    message_type=removed.type,
                    name=removed.name,
                    stream_type=stream_type,
                    is_reasoning=is_reasoning,
                    section=_text(removed.metadata.get("section")),
                ),
            )

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
            was_streamed = message_id in tracker.streamed_message_ids
            previous_output = existing.output if was_streamed else None
            output = token if payload.get("isSequence") is True else f"{existing.output}{token}"
            updated_message = ThinkMessage(
                id=existing.id,
                run_id=existing.run_id,
                type=existing.type,
                output=output,
                name=existing.name,
                metadata=existing.metadata,
                is_error=existing.is_error,
            )
            tracker.messages[message_id] = updated_message
            self._recompute_final_locked(tracker)
            self._refresh_needs_input_prompt_locked(tracker)
            completed = self._mark_succeeded_locked(tracker, require_task_end=True)
            final_output = tracker.final_output
            tracker.condition.notify_all()
        self._record_stream_update(
            tracker,
            updated_message,
            previous_output=previous_output,
        )
        if completed:
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)

    def _latest_assistant_prompt(self, tracker: _RunTracker) -> str | None:
        prompt: str | None = None
        for message_id in tracker.message_order:
            message = tracker.messages[message_id]
            if message.type == "user_message":
                prompt = None
                continue
            if (
                message.type == "assistant_message"
                and not _stream_traits(message)[1]
                and _progress_update(message) is None
                and message.output.strip()
            ):
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
                if tracker.cancel_requested:
                    tracker.status = RunStatus.CANCELLING
                elif tracker.final_output is not None and tracker.task_ended_at is not None:
                    self._mark_succeeded_locked(tracker, require_task_end=True)
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
                    if message.type == "user_message":
                        with tracker.condition:
                            tracker.options = _run_options_from_metadata(
                                message.metadata,
                                tracker.options,
                            )
        completed = False
        cancelled = False
        resumed_running = False
        final_output: str | None = None
        with tracker.condition:
            was_awaiting_transcript = tracker.awaiting_transcript
            tracker.awaiting_transcript = False
            self._refresh_needs_input_prompt_locked(tracker)
            durable_stop = self._latest_turn_was_manually_stopped_locked(tracker)
            if tracker.failure is not None:
                tracker.status = RunStatus.FAILED
            elif durable_stop:
                cancelled = tracker.status is not RunStatus.CANCELLED
                tracker.status = RunStatus.CANCELLED
                tracker.stop_released = True
                tracker.needs_input = None
                tracker.failure = RunCancelledError(
                    f"Think run {tracker.run_id or ''} was cancelled"
                )
            elif tracker.cancel_requested:
                tracker.status = RunStatus.CANCELLING
            elif tracker.final_output is not None:
                completed = self._mark_succeeded_locked(tracker, require_task_end=False)
            elif tracker.status in {RunStatus.SUBMITTING, RunStatus.SUCCEEDED}:
                tracker.status = RunStatus.QUEUED
            resumed_running = (
                was_awaiting_transcript
                and tracker.needs_input is None
                and tracker.status in {RunStatus.QUEUED, RunStatus.RUNNING}
            )
            final_output = tracker.final_output
            tracker.condition.notify_all()
        if cancelled:
            self._record_event(
                tracker,
                EventKind.CANCELLED,
                message="Analysis cancelled",
                data={"durable": True},
            )
        elif completed:
            self._record_event(tracker, EventKind.COMPLETED, message=final_output)
        elif resumed_running:
            self._record_event(
                tracker,
                EventKind.STARTED,
                message="Analysis resumed",
                data={"resumed": True},
            )

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

    def _emit_stop_request(
        self,
        tracker: _RunTracker,
        *,
        raise_on_error: bool = False,
    ) -> None:
        with tracker.condition:
            run_id = tracker.run_id
            task_id = tracker.task_id
            client_observed_active = tracker.status in {
                RunStatus.SUBMITTING,
                RunStatus.QUEUED,
                RunStatus.RUNNING,
                RunStatus.CANCELLING,
            }
        if run_id is None:
            raise RunProtocolError("Cannot cancel a run without an id")
        payload: dict[str, object] = {
            "threadId": run_id,
            "clientObservedActive": client_observed_active,
        }
        if task_id is not None:
            payload["taskId"] = task_id
        try:
            self._socket.emit("stop", payload)
        except (SocketIOError, OSError) as exc:
            if raise_on_error:
                raise ThinkConnectionError(
                    "Could not request Think cancellation"
                ) from exc
            logger.warning(
                "Think cancellation request failed; reconnect will retry",
                exc_info=True,
            )
        except Exception:
            logger.warning(
                "Unexpected Think cancellation request failure; reconnect will retry",
                exc_info=True,
            )
            if raise_on_error:
                raise

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

    @staticmethod
    def _billing_retry_payload(payload: Mapping[str, object]) -> dict[str, object]:
        retry_payload = dict(payload)
        message = _mapping(retry_payload.get("message"))
        metadata = _mapping(message.get("metadata"))
        metadata["billingRetrySalt"] = str(uuid.uuid4())
        message["metadata"] = metadata
        retry_payload["message"] = message
        return retry_payload

    def _apply_billing_top_up(
        self,
        request: BillingTopUpRequest,
        approval: BillingTopUpApproval,
    ) -> None:
        if approval.amount_cents < request.minimum_top_up_cents:
            raise ValueError(
                "approved top-up amount cannot be smaller than the required amount"
            )
        try:
            response = self._session.put(
                f"{self.think_url}/user/billing/spend-cap",
                json={
                    "mode": "fixed",
                    "monthlyLimitUsd": (
                        f"{approval.amount_cents // 100}."
                        f"{approval.amount_cents % 100:02d}"
                    ),
                    "topUp": True,
                },
                timeout=self._http_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ThinkConnectionError("Could not apply the approved billing top-up") from exc
        except Exception:
            logger.exception("Unexpected Think billing top-up failure")
            raise
        payload = _raise_for_response(response, action="billing top-up")
        if payload.get("spendCapSaved") is True:
            return
        payment_setup = payload.get("paymentMethodSetupRequired") is True
        tax_setup = payload.get("taxLocationSetupRequired") is True
        action_url = _text(payload.get("checkoutUrl")) or _text(payload.get("portalUrl"))
        message = (
            "A payment method must be set up before the approved top-up can be applied."
            if payment_setup
            else "Billing details must be updated before the approved top-up can be applied."
            if tax_setup
            else "Think did not confirm that the approved top-up was saved."
        )
        raise ThinkBillingRequiredError(
            message,
            status_code=402,
            code=(
                "payment_method_setup_required"
                if payment_setup
                else "tax_location_setup_required"
                if tax_setup
                else "billing_required"
            ),
            request=request,
            action_url=action_url,
        )

    def _send_client_message_with_billing_approval(
        self,
        *,
        payload: Mapping[str, object],
        run_id: str | None,
    ) -> dict[str, object]:
        try:
            return self._send_client_message(payload=payload, run_id=run_id)
        except RunRejectedError as error:
            if not _is_billing_rejection(error):
                raise
            billing_error = _billing_required_error(error)
            rejection = error
        request = billing_error.request
        callback = self._on_billing_required
        if request is None or callback is None:
            raise billing_error from rejection
        approval = callback(request)
        if approval is None:
            raise billing_error from rejection
        if not isinstance(approval, BillingTopUpApproval):
            raise TypeError(
                "on_billing_required must return request.approve(...) or None"
            )
        self._apply_billing_top_up(request, approval)
        retry_payload = self._billing_retry_payload(payload)
        try:
            return self._send_client_message(payload=retry_payload, run_id=run_id)
        except RunRejectedError as retry_error:
            if _is_billing_rejection(retry_error):
                raise _billing_required_error(retry_error) from retry_error
            raise

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
        files: FileInputs,
        on_upload_progress: UploadProgressCallback | None,
    ) -> list[UploadedFile]:
        resolved: list[UploadedFile] = []
        items = (
            (files,)
            if isinstance(files, (UploadedFile, str, Path))
            else files
        )
        for item in items:
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
        files: FileInputs = (),
        datasets: Iterable[Dataset] = (),
        options: RunOptions | None = None,
        on_upload_progress: UploadProgressCallback | None = None,
        idempotency_key: str | None = None,
    ) -> "Run":
        """Upload inputs and submit a new Think workload."""

        return self._submit(
            prompt,
            files=files,
            datasets=datasets,
            options=options,
            on_upload_progress=on_upload_progress,
            progress_callback=None,
            idempotency_key=idempotency_key,
        )

    def submit_with_progress(
        self,
        prompt: MessageInput,
        *,
        files: FileInputs = (),
        datasets: Iterable[Dataset] = (),
        options: RunOptions | None = None,
        on_upload_progress: UploadProgressCallback | None = None,
        on_event: ProgressCallback | None = None,
        idempotency_key: str | None = None,
    ) -> "Run":
        """Submit and render lifecycle plus browser-visible streamed output."""

        return self._submit(
            prompt,
            files=files,
            datasets=datasets,
            options=options,
            on_upload_progress=on_upload_progress,
            progress_callback=on_event or self._on_event or ProgressRenderer(),
            idempotency_key=idempotency_key,
        )

    def _submit(
        self,
        prompt: MessageInput,
        *,
        files: FileInputs,
        datasets: Iterable[Dataset],
        options: RunOptions | None,
        on_upload_progress: UploadProgressCallback | None,
        progress_callback: ProgressCallback | None,
        idempotency_key: str | None,
    ) -> "Run":
        message = prompt if isinstance(prompt, MessageWithContext) else MessageWithContext(prompt)
        normalized_prompt = message.prompt.strip()
        message_id = _message_id(idempotency_key, run_id=None)
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
        tracker = _RunTracker(
            run_id=None,
            options=resolved_options,
            progress_callback=progress_callback,
        )
        with self._state_lock:
            self._active_tracker = tracker
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
            ack = self._send_client_message_with_billing_approval(
                payload=payload,
                run_id=None,
            )
            run_id = _text(ack.get("threadId"))
            if run_id is None:
                raise RunProtocolError("Think accepted the workload without returning a run id")
            replayed = ack.get("dispatched") is False
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
            tracker.awaiting_transcript = replayed
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
        if replayed:
            self._refresh(tracker)
        else:
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
        try:
            was_connected = self._socket.connected
            self.connect()
            if was_connected:
                self._room_sync(normalized_id)
                self._socket.emit("connection_successful")
                self._emit_status_ready(normalized_id)
        except Exception as exc:
            logger.debug("Think resume failed before transcript hydration", exc_info=True)
            with tracker.condition:
                tracker.failure = exc
                tracker.status = RunStatus.FAILED
                tracker.awaiting_transcript = False
                tracker.condition.notify_all()
            raise
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
        files: FileInputs,
        datasets: Iterable[Dataset],
        planning_response: bool,
        on_upload_progress: UploadProgressCallback | None,
        idempotency_key: str | None,
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
            prior_final_result = tracker.final_result
            prior_failure = tracker.failure
            prior_task_ended_at = tracker.task_ended_at
            prior_task_id = tracker.task_id
            prior_cancel_requested = tracker.cancel_requested
            prior_cancelling_event_emitted = tracker.cancelling_event_emitted
            prior_stop_released = tracker.stop_released
            prior_refresh_requested = tracker.refresh_requested
            prior_answered_checkpoint_id = tracker.answered_checkpoint_id
            if run_id is None:
                raise RunProtocolError("Cannot continue a run without an id")
            message_id = _message_id(idempotency_key, run_id=run_id)
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
            tracker.final_result = None
            tracker.failure = None
            tracker.task_ended_at = None
            tracker.task_id = None
            tracker.cancel_requested = False
            tracker.cancelling_event_emitted = False
            tracker.stop_released = False
            tracker.refresh_requested = False
            tracker.status = RunStatus.SUBMITTING
            tracker.condition.notify_all()
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
            ack = self._send_client_message_with_billing_approval(
                payload=payload,
                run_id=run_id,
            )
        except Exception:
            logger.debug("Think continuation failed before acknowledgement", exc_info=True)
            with tracker.condition:
                if tracker.status is RunStatus.SUBMITTING:
                    tracker.status = prior_status
                    tracker.needs_input = prior_input
                    tracker.final_output = prior_final_output
                    tracker.final_message_id = prior_final_message_id
                    tracker.final_result = prior_final_result
                    tracker.failure = prior_failure
                    tracker.task_ended_at = prior_task_ended_at
                    tracker.task_id = prior_task_id
                    tracker.cancel_requested = prior_cancel_requested
                    tracker.cancelling_event_emitted = prior_cancelling_event_emitted
                    tracker.stop_released = prior_stop_released
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
        with self._state_lock:
            if tracker is not self._active_tracker:
                raise InputResponseError("This run is no longer active on its client")
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

    def list_conversations(
        self,
        *,
        search: str | None = None,
        page_size: int = 50,
        limit: int | None = None,
    ) -> tuple[PreviousConversation, ...]:
        """List the authenticated user's conversations, newest first."""

        if not 1 <= page_size <= 1_000:
            raise ValueError("page_size must be between 1 and 1000")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        self.connect()
        conversations: list[PreviousConversation] = []
        cursor: str | None = None
        normalized_search = _text(search)
        while True:
            request_size = (
                page_size
                if limit is None
                else min(page_size, limit - len(conversations))
            )
            try:
                response = self._session.post(
                    f"{self.think_url}/project/threads",
                    json={
                        "pagination": {"first": request_size, "cursor": cursor},
                        "filter": {"search": normalized_search},
                    },
                    timeout=self._http_timeout,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise ThinkConnectionError("Could not list Think conversations") from exc
            try:
                payload = _raise_for_response(response, action="conversation listing")
            finally:
                try:
                    response.close()
                except Exception:
                    logger.warning("Think conversation listing cleanup failed", exc_info=True)
            raw_conversations = payload.get("data")
            page_info = _mapping(payload.get("pageInfo"))
            has_next_page = page_info.get("hasNextPage")
            if not isinstance(raw_conversations, list) or not isinstance(has_next_page, bool):
                raise RunProtocolError("Think returned an invalid conversation listing")
            for raw_conversation in cast(list[object], raw_conversations):
                conversation = _mapping(raw_conversation)
                conversation_id = _text(conversation.get("id"))
                if conversation_id is None:
                    raise RunProtocolError("Think conversation listing omitted an id")
                conversations.append(
                    PreviousConversation(
                        conversation_id,
                        _text(conversation.get("name")),
                        _optional_utc_datetime(
                            conversation.get("createdAt"),
                            field_name="conversation creation timestamp",
                        ),
                    )
                )
                if limit is not None and len(conversations) >= limit:
                    return tuple(conversations)
            if not has_next_page:
                return tuple(conversations)
            next_cursor = _text(page_info.get("endCursor"))
            if next_cursor is None or next_cursor == cursor:
                raise RunProtocolError("Think conversation pagination did not advance")
            cursor = next_cursor

    def _list_output_files(
        self,
        run_id: str,
        *,
        page_size: int,
    ) -> tuple[OutputFile, ...]:
        if not 1 <= page_size <= 1_000:
            raise ValueError("page_size must be between 1 and 1000")
        self.connect()
        files: list[OutputFile] = []
        offset = 0
        while True:
            try:
                response = self._session.get(
                    f"{self.think_url}/api/user-output/list",
                    params={"thread_id": run_id, "limit": page_size, "offset": offset},
                    timeout=self._http_timeout,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise ThinkConnectionError("Could not list Think output files") from exc
            try:
                payload = _raise_for_response(response, action="output listing")
            finally:
                try:
                    response.close()
                except Exception:
                    logger.warning("Think output listing cleanup failed", exc_info=True)
            raw_files = payload.get("files")
            has_more = payload.get("hasMore")
            if not isinstance(raw_files, list) or not isinstance(has_more, bool):
                raise RunProtocolError("Think returned an invalid output listing")
            page = [_output_file(_mapping(item)) for item in cast(list[object], raw_files)]
            files.extend(page)
            if not has_more:
                return tuple(files)
            if not page:
                raise RunProtocolError("Think output pagination did not advance")
            offset += len(page)

    def _download_output(
        self,
        *,
        route: str,
        run_id: str,
        destination: str | Path | None,
        default_name: str,
        overwrite: bool,
        chunk_size: int,
    ) -> Path:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        target = Path(destination).expanduser() if destination is not None else Path(default_name)
        if target.is_dir():
            target /= default_name
        if target.exists() and not overwrite:
            raise FileExistsError(f"download destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.connect()
        try:
            response = self._session.get(
                f"{self.think_url}{route}",
                params={"thread_id": run_id},
                timeout=self._http_timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise ThinkConnectionError("Could not download Think output") from exc
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        completed = False
        try:
            if not 200 <= response.status_code < 300:
                _raise_for_response(response, action="output download")
            try:
                with temporary.open("xb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            except requests.RequestException as exc:
                raise ThinkConnectionError("Think output download was interrupted") from exc
            if overwrite:
                temporary.replace(target)
            else:
                target.hardlink_to(temporary)
                temporary.unlink()
            completed = True
            return target
        finally:
            try:
                response.close()
            except Exception:
                logger.warning("Think download response cleanup failed", exc_info=True)
            if not completed:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Think partial download cleanup failed", exc_info=True)
                except Exception:
                    logger.warning("Unexpected Think download cleanup failure", exc_info=True)

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
            with self._state_lock:
                tracker = self._active_tracker
            if tracker is None:
                tracker_callback = None
                run_id = None
            else:
                with tracker.condition:
                    tracker_callback = tracker.progress_callback
                    run_id = tracker.run_id
            if self._socket.connected:
                try:
                    self._socket.disconnect()
                except (SocketIOError, OSError):
                    logger.debug("Think socket disconnect failed", exc_info=True)
                except Exception:
                    logger.warning("Unexpected Think socket disconnect failure", exc_info=True)
            close_progress_callback(tracker_callback, run_id)
            if self._on_event is not tracker_callback:
                close_progress_callback(self._on_event, run_id)
            if self._owns_session:
                self._session.close()


class Run:
    """Handle to a durable Think conversation and its current turn."""

    def __init__(self, client: ThinkClient, tracker: _RunTracker) -> None:
        self._client = client
        self._tracker = tracker

    @property
    def id(self) -> str:  # noqa: A003 - concise public run identifier
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
                if not _stream_traits(self._tracker.messages[message_id])[1]
                and _progress_update(self._tracker.messages[message_id]) is None
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
            if self._tracker.final_result is None:
                final_message_id = self._tracker.final_message_id
                if final_message_id is None:
                    raise RunProtocolError("Run succeeded without a final message")
                final_message = self._tracker.messages[final_message_id]
                metadata = final_message.metadata
                (
                    execution_started_at,
                    execution_completed_at,
                    execution_duration_seconds,
                ) = _result_execution_metadata(metadata)
                self._tracker.final_result = RunResult(
                    run_id=self.id,
                    output=self._tracker.final_output,
                    options=self._tracker.options,
                    execution_started_at=execution_started_at,
                    execution_completed_at=execution_completed_at,
                    execution_duration_seconds=execution_duration_seconds,
                    _file_loader=self.output_files,
                )
            return self._tracker.final_result
        return None

    def _terminal_state_locked(
        self,
    ) -> tuple[RunOutcome | None, Exception | None, bool]:
        try:
            outcome = self._outcome_locked()
        except RunProtocolError as exc:
            self._tracker.failure = exc
            self._tracker.status = RunStatus.FAILED
            self._tracker.condition.notify_all()
            return None, exc, False
        failure = self._tracker.failure
        should_refresh = False
        if (
            failure is None
            and outcome is None
            and not self._tracker.cancel_requested
            and self._tracker.task_ended_at is not None
        ):
            elapsed = self._client._clock() - self._tracker.task_ended_at
            if elapsed >= 1.0 and not self._tracker.refresh_requested:
                self._tracker.refresh_requested = True
                should_refresh = True
            elif elapsed >= self._client._finalization_timeout:
                latest_error = self._client._latest_turn_error_locked(self._tracker)
                if latest_error is None:
                    failure = RunProtocolError(
                        "Think ended the task without a final response or "
                        "needs_input state"
                    )
                else:
                    detail = latest_error.output.strip() or "Server execution failed"
                    failure = RunFailedError(
                        f"Think run {latest_error.run_id} failed: {detail}",
                        run_id=latest_error.run_id,
                        message_id=latest_error.id,
                    )
                self._tracker.failure = failure
                self._tracker.status = RunStatus.FAILED
                self._tracker.condition.notify_all()
                if latest_error is not None:
                    self._client._record_event(
                        self._tracker,
                        EventKind.FAILED,
                        message=latest_error.output.strip()
                        or "Server execution failed",
                        data={"message_id": latest_error.id, "durable": True},
                    )
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
            remaining: float | None = None
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

    async def aevents(
        self,
        timeout: float | None = None,
        *,
        poll_interval: float = 0.1,
    ) -> AsyncIterator[ThinkEvent]:
        """Asynchronously yield ordered events without blocking the event loop."""

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        deadline = None if timeout is None else self._client._clock() + timeout
        cursor = 0
        pending_events: deque[ThinkEvent] = deque()
        while True:
            event: ThinkEvent | None = None
            outcome: RunOutcome | None = None
            failure: Exception | None = None
            should_refresh = False
            remaining: float | None = None
            with self._tracker.condition:
                if not pending_events:
                    pending_events.extend(self._events_after_locked(cursor))
                if pending_events:
                    event = pending_events.popleft()
                    cursor = event.sequence
                else:
                    outcome, failure, should_refresh = self._terminal_state_locked()
                    remaining = (
                        None
                        if deadline is None
                        else deadline - self._client._clock()
                    )
            if failure is not None:
                raise failure
            if event is not None:
                yield event
                continue
            if outcome is not None:
                return
            if should_refresh:
                await asyncio.to_thread(self.refresh)
                continue
            if remaining is not None and remaining <= 0:
                raise RunTimeoutError(f"Timed out streaming Think run {self.id}")
            await asyncio.sleep(
                poll_interval
                if remaining is None
                else min(poll_interval, remaining)
            )

    async def await_result(self, timeout: float | None = None) -> RunOutcome:
        """Await the current turn while keeping an asyncio loop responsive."""

        async for _event in self.aevents(timeout=timeout):
            pass
        with self._tracker.condition:
            outcome, failure, _should_refresh = self._terminal_state_locked()
        if failure is not None:
            raise failure
        if outcome is None:
            raise RunProtocolError("Think event stream ended without an outcome")
        return outcome

    def interact(
        self,
        timeout: float | None = None,
        *,
        on_clarification: InputCallback | None = None,
        on_plan_review: InputCallback | None = None,
    ) -> RunOutcome:
        """Drive clarification and plan-review pauses until the turn completes.

        Missing callbacks use a terminal/notebook input prompt. Billing pauses
        remain explicit and are returned unchanged for dashboard resolution.
        """

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        deadline = None if timeout is None else self._client._clock() + timeout
        while True:
            remaining = (
                None if deadline is None else deadline - self._client._clock()
            )
            if remaining is not None and remaining <= 0:
                raise RunTimeoutError(f"Timed out interacting with Think run {self.id}")
            outcome = self.wait(timeout=remaining)
            if isinstance(outcome, RunResult) or outcome.kind is InputKind.BILLING:
                return outcome
            callback = (
                on_plan_review
                if outcome.kind is InputKind.PLAN_REVIEW
                else on_clarification
            )
            if callback is None:
                prompt = outcome.prompt or f"Input required ({outcome.kind.value})"
                response: MessageInput = input(f"{prompt}\n> ")
            else:
                response = callback(outcome)
            self.respond(response)

    def respond(
        self,
        response: MessageInput,
        *,
        files: FileInputs = (),
        datasets: Iterable[Dataset] = (),
        on_upload_progress: UploadProgressCallback | None = None,
        idempotency_key: str | None = None,
    ) -> Self:
        """Answer a clarification or approve/revise a proposed plan."""

        self._client._continue(
            self._tracker,
            response,
            files=files,
            datasets=datasets,
            planning_response=True,
            on_upload_progress=on_upload_progress,
            idempotency_key=idempotency_key,
        )
        return self

    def follow_up(
        self,
        prompt: MessageInput,
        *,
        files: FileInputs = (),
        datasets: Iterable[Dataset] = (),
        on_upload_progress: UploadProgressCallback | None = None,
        idempotency_key: str | None = None,
    ) -> Self:
        """Start another turn in a successfully completed conversation."""

        self._client._continue(
            self._tracker,
            prompt,
            files=files,
            datasets=datasets,
            planning_response=False,
            on_upload_progress=on_upload_progress,
            idempotency_key=idempotency_key,
        )
        return self

    def cancel(self, timeout: float = 60.0) -> Self:
        """Cancel durable server work and wait for stop cleanup to be released."""

        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._tracker.condition:
            status = self._tracker.status
            if status is RunStatus.CANCELLED:
                return self
            if status in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
                raise InputResponseError("A terminal run cannot be cancelled")
            if status is RunStatus.NEEDS_INPUT:
                raise InputResponseError(
                    "A paused run has no active work to cancel; respond or detach instead"
                )
            self._tracker.cancel_requested = True
            self._tracker.status = RunStatus.CANCELLING
            should_emit = not self._tracker.cancelling_event_emitted
            self._tracker.cancelling_event_emitted = True
            self._tracker.condition.notify_all()
        if should_emit:
            self._client._record_event(
                self._tracker,
                EventKind.CANCELLING,
                message="Cancelling analysis",
            )
        self._client.connect()
        self._client._emit_stop_request(self._tracker, raise_on_error=True)
        deadline = self._client._clock() + timeout
        with self._tracker.condition:
            while self._tracker.status is not RunStatus.CANCELLED:
                failure = self._tracker.failure
                if failure is not None and not isinstance(
                    failure,
                    RunCancelledError,
                ):
                    raise failure
                remaining = deadline - self._client._clock()
                if remaining <= 0:
                    raise RunTimeoutError(
                        f"Timed out cancelling Think run {self.id}; "
                        "the request will be retried after reconnect"
                    )
                self._tracker.condition.wait(timeout=remaining)
        return self

    def detach(self) -> None:
        """Disconnect locally while leaving durable server work running."""

        self._client.close()

    def refresh(self) -> Self:
        """Rehydrate durable transcript and pause state after reconnecting."""

        self._client._refresh(self._tracker)
        return self

    def output_files(self, *, page_size: int = 500) -> tuple[OutputFile, ...]:
        """Return every generated output file, transparently paginating."""

        return self._client._list_output_files(self.id, page_size=page_size)

    def download_file(
        self,
        output: OutputFile | str,
        destination: str | Path | None = None,
        *,
        overwrite: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        """Stream one generated file to disk through the authenticated session."""

        output_path = _normalize_output_path(output.path if isinstance(output, OutputFile) else output)
        return self._client._download_output(
            route=f"/api/user-output/download/{quote(output_path, safe='/')}",
            run_id=self.id,
            destination=destination,
            default_name=output_path.rsplit("/", maxsplit=1)[-1],
            overwrite=overwrite,
            chunk_size=chunk_size,
        )

    def download_all(
        self,
        destination: str | Path | None = None,
        *,
        overwrite: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        """Stream all generated files to an uncompressed tar archive."""

        return self._client._download_output(
            route="/api/user-output/download-all",
            run_id=self.id,
            destination=destination,
            default_name=f"output-{self.id}.tar",
            overwrite=overwrite,
            chunk_size=chunk_size,
        )


__all__ = [
    "DEFAULT_THINK_URL",
    "ProgressCallback",
    "Run",
    "show_progress",
    "ThinkClient",
    "UploadProgressCallback",
]
