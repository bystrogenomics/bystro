from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import time
from typing import cast

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import requests
import socketio

from bystro.api import auth as dashboard_auth
from bystro.api.auth import CachedAuth
import bystro.think.client as think_client_module
from bystro.think import (
    Dataset,
    EventKind,
    InputKind,
    InputResponseError,
    MessageWithContext,
    NeedsInput,
    OutputFile,
    PreviousConversation,
    ProgressPhase,
    ProgressUpdate,
    ProgressRenderer,
    RunCancelledError,
    RunOptions,
    RunProtocolError,
    RunRejectedError,
    RunResult,
    RunStatus,
    RunTimeoutError,
    StreamUpdate,
    ThinkClient,
    ThinkAuthenticationError,
    ThinkEvent,
    ThinkConnectionError,
    ThinkHTTPError,
    UploadedFile,
    UploadPhase,
    UploadProgress,
    add_artifact_context,
    add_genetic_context,
    add_previous_conversation_context,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object,
        *,
        chunks: tuple[bytes, ...] = (),
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._chunks = chunks
        self.reason = "OK" if status_code < 400 else "Error"
        self.chunk_sizes: list[int] = []
        self.closed = False

    def json(self) -> object:
        return self._payload

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        self.chunk_sizes.append(chunk_size)
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class InvalidJSONResponse(FakeResponse):
    def json(self) -> object:
        raise ValueError("not JSON")


class FakeSession:
    def __init__(self) -> None:
        self.cookies = requests.cookies.RequestsCookieJar()
        self.posts: list[tuple[str, dict[str, object]]] = []
        self.gets: list[tuple[str, dict[str, object]]] = []
        self.upload_responses: list[FakeResponse] = []
        self.status_responses: list[FakeResponse] = []
        self.artifact_responses: dict[str, FakeResponse] = {}
        self.conversation_responses: list[FakeResponse] = []
        self.output_responses: list[FakeResponse] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.posts.append((url, kwargs))
        if url.endswith("/auth/cookie"):
            self.cookies.set("access_token", "chainlit-session", path="/")
            return FakeResponse(200, {"success": True})
        if url.endswith("/set-session-cookie"):
            return FakeResponse(200, {"message": "Session cookie set"})
        if url.endswith("/user/files/chunk"):
            return self.upload_responses.pop(0)
        if url.endswith("/project/threads"):
            return self.conversation_responses.pop(0)
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.gets.append((url, kwargs))
        if "/api/user-output/" in url:
            return self.output_responses.pop(0)
        if "/user/files/chunk/status" in url:
            return self.status_responses.pop(0)
        if "/user/files/" in url:
            artifact_id = url.rsplit("/", maxsplit=1)[-1]
            return self.artifact_responses[artifact_id]
        raise AssertionError(f"unexpected GET {url}")

    def close(self) -> None:
        return None


class FakeSocket:
    def __init__(self) -> None:
        self.handlers: dict[str, Callable[..., object]] = {}
        self.calls: list[tuple[str, object, float | None]] = []
        self.emits: list[tuple[str, object]] = []
        self.connect_args: tuple[str, dict[str, object]] | None = None
        self.connected = False
        self.connect_failure: Exception | None = None
        self.next_message_ack: dict[str, object] = {
            "success": True,
            "threadId": "thread-1",
            "created": True,
            "dispatched": True,
        }
        self.call_hook: Callable[[str, object, float | None], None] | None = None
        self.emit_hook: Callable[[str, object], None] | None = None

    def on(
        self,
        event: str,
        handler: Callable[..., object] | None = None,
        namespace: str | None = None,
    ) -> Callable[..., object]:
        del namespace

        def register(callback: Callable[..., object]) -> Callable[..., object]:
            self.handlers[event] = callback
            return callback

        if handler is not None:
            return register(handler)
        return register

    def connect(self, url: str, **kwargs: object) -> None:
        if self.connect_failure is not None:
            raise self.connect_failure
        self.connect_args = (url, kwargs)
        self.connected = True
        callback = self.handlers.get("connect")
        if callback is not None:
            callback()

    def call(
        self,
        event: str,
        data: object = None,
        timeout: float | None = None,
    ) -> object:
        self.calls.append((event, data, timeout))
        if self.call_hook is not None:
            self.call_hook(event, data, timeout)
        if event == "thread_detach":
            return {"success": True}
        if event == "thread_change":
            assert isinstance(data, dict)
            return {"success": True, "threadId": data["threadId"]}
        if event == "client_message":
            return dict(self.next_message_ack)
        if event == "action":
            return {"success": True}
        raise AssertionError(f"unexpected socket call {event}")

    def emit(self, event: str, data: object = None) -> None:
        self.emits.append((event, data))
        if self.emit_hook is not None:
            self.emit_hook(event, data)

    def disconnect(self) -> None:
        self.connected = False

    def trigger(self, event: str, payload: object = None) -> object:
        callback = self.handlers[event]
        if payload is None:
            return callback()
        return callback(payload)


@pytest.fixture
def auth_state() -> CachedAuth:
    return CachedAuth(
        email="scientist@example.com",
        access_token="dashboard-jwt",
        url="https://bystro.cloud",
    )


@pytest.fixture
def transport() -> tuple[FakeSession, FakeSocket]:
    return FakeSession(), FakeSocket()


def make_client(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    *,
    on_event: Callable[[ThinkEvent], None] | None = None,
    upload_chunk_size: int = 10 * 1024 * 1024,
    sleep: Callable[[float], None] = time.sleep,
    finalization_timeout: float = 10.0,
    transports: tuple[str, ...] | None = None,
) -> ThinkClient:
    session, socket = transport
    return ThinkClient(
        auth=auth_state,
        think_url="https://ai.bystro.cloud",
        on_event=on_event,
        _session=session,
        _socket_factory=lambda **_kwargs: socket,
        upload_chunk_size=upload_chunk_size,
        finalization_timeout=finalization_timeout,
        transports=transports,
        _sleep=sleep,
    )


def test_login_forwards_site_access_and_legal_consent(
    monkeypatch,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_dashboard_login(
        email: str,
        password: str,
        **kwargs: object,
    ) -> CachedAuth:
        calls.append((email, password, kwargs))
        return auth_state

    monkeypatch.setattr(dashboard_auth, "login", fake_dashboard_login)
    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(socketio, "Client", lambda **_kwargs: socket)
    consent = dashboard_auth.LegalConsent.accepted("Scientist Example")

    client = ThinkClient.login(
        "scientist@example.com",
        "password",
        dashboard_url="https://example.test",
        site_access_code="invite-code",
        legal_consent=consent,
        cache=False,
    )

    assert calls == [
        (
            "scientist@example.com",
            "password",
            {
                "host": "https://example.test",
                "cache": False,
                "site_access_code": "invite-code",
                "legal_consent": consent,
            },
        )
    ]
    client.close()


def test_connect_bootstraps_existing_browser_auth_contract(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    client = make_client(auth_state, transport)

    client.connect()

    assert [url for url, _ in session.posts[:2]] == [
        "https://ai.bystro.cloud/auth/cookie",
        "https://ai.bystro.cloud/set-session-cookie",
    ]
    assert all(options["allow_redirects"] is False for _, options in session.posts[:2])
    assert session.posts[1][1]["json"] == {"session_id": client.session_id}
    assert session.cookies.get("bystro_access_token", domain="ai.bystro.cloud") == "dashboard-jwt"
    assert socket.connect_args is not None
    socket_url, socket_kwargs = socket.connect_args
    assert socket_url == "https://ai.bystro.cloud"
    assert socket_kwargs["socketio_path"] == "ws/socket.io"
    auth_callback = socket_kwargs["auth"]
    assert callable(auth_callback)
    auth_payload = auth_callback()
    assert auth_payload == {
        "clientType": "webapp",
        "sessionId": client.session_id,
        "threadId": "",
        "userEnv": "{}",
        "chatProfile": "",
    }
    assert ("connection_successful", None) in socket.emits


def test_connect_can_require_websocket_transport(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport, transports=("websocket",))

    client.connect()

    assert socket.connect_args is not None
    _socket_url, socket_kwargs = socket.connect_args
    assert socket_kwargs["transports"] == ["websocket"]


def test_connect_accepts_a_single_transport_string(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    client = ThinkClient(
        auth=auth_state,
        think_url="https://ai.bystro.cloud",
        _session=session,
        _socket_factory=lambda **_kwargs: socket,
        transports="websocket",
    )

    client.connect()

    assert socket.connect_args is not None
    _socket_url, socket_kwargs = socket.connect_args
    assert socket_kwargs["transports"] == ["websocket"]


def test_connect_rejects_unknown_or_empty_transports(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    for transports in ((), ("websocket", "invalid")):
        with pytest.raises(ValueError, match="transport"):
            ThinkClient(
                auth=auth_state,
                _session=session,
                _socket_factory=lambda **_kwargs: socket,
                transports=transports,
            )


def test_submit_reports_connected_before_submitted(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    events: list[ThinkEvent] = []
    client = make_client(auth_state, transport, on_event=events.append)

    run = client.submit("Analyze")

    assert [(event.kind, event.run_id) for event in events[:2]] == [
        (EventKind.CONNECTED, run.id),
        (EventKind.SUBMITTED, run.id),
    ]


def test_submit_with_progress_is_the_canonical_concise_renderer(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)

    run = client.submit_with_progress("Analyze a secret prompt")
    socket.trigger("task_start", {"threadId": run.id})
    socket.trigger("status_overlay_update", {"threadId": run.id, "display": "progress"})
    socket.trigger(
        "status_overlay_update",
        {"threadId": run.id, "visible": True, "display": "Processing..."},
    )
    socket.trigger("disconnect", "client disconnect")

    output = capsys.readouterr().out
    assert "secret prompt" not in output
    assert output.splitlines() == [
        "[connected] Connected to Think",
        "[submitted] Workload submitted",
        "[started] Analysis started",
        "[progress] Processing...",
    ]


def test_progress_renderer_coalesces_repeated_overlay_snapshots(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)

    run = client.submit_with_progress("Analyze")
    for _ in range(4):
        socket.trigger(
            "status_overlay_update",
            {
                "threadId": run.id,
                "visible": True,
                "display": "Processing...",
            },
        )

    output = capsys.readouterr().out
    assert output.count("[progress] Processing...") == 1


def test_progress_renderer_coalesces_alternating_generic_states() -> None:
    output = StringIO()
    renderer = ProgressRenderer(file=output, heartbeat_interval=30.0)
    started_at = datetime(2026, 8, 15, tzinfo=timezone.utc)

    renderer(ThinkEvent(1, EventKind.SUBMITTED, "thread-1", started_at))
    for sequence in (2, 4):
        renderer(
            ThinkEvent(
                sequence,
                EventKind.STREAM,
                "thread-1",
                started_at + timedelta(seconds=sequence),
                stream_update=StreamUpdate(
                    message_id=f"reasoning-{sequence}",
                    delta="",
                    operation="append",
                    content_length=10,
                    message_type="assistant_message",
                    is_reasoning=True,
                ),
            )
        )
        renderer(
            ThinkEvent(
                sequence + 1,
                EventKind.PROGRESS,
                "thread-1",
                started_at + timedelta(seconds=sequence + 1),
                message="Processing...",
            )
        )

    rendered = output.getvalue()
    assert rendered.count("[progress] Thinking...") == 1
    assert rendered.count("[progress] Processing...") == 1


def test_progress_renderer_turns_repeated_processing_into_elapsed_heartbeats() -> None:
    output = StringIO()
    renderer = ProgressRenderer(file=output, heartbeat_interval=30.0)
    started_at = datetime(2026, 8, 15, tzinfo=timezone.utc)

    renderer(
        ThinkEvent(
            1,
            EventKind.SUBMITTED,
            "thread-1",
            started_at,
        )
    )
    for sequence, seconds in ((2, 1), (3, 15), (4, 31), (5, 45), (6, 61)):
        renderer(
            ThinkEvent(
                sequence,
                EventKind.PROGRESS,
                "thread-1",
                started_at + timedelta(seconds=seconds),
                message="Processing...",
            )
        )

    assert output.getvalue().splitlines() == [
        "[submitted] Workload submitted",
        "[progress] Processing...",
        "[progress] Still working... (31s elapsed)",
        "[progress] Still working... (1m 1s elapsed)",
    ]


def test_structured_backend_progress_is_typed_rendered_and_not_transcript(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    output = StringIO()
    renderer = ProgressRenderer(file=output)
    events: list[ThinkEvent] = []
    _session, socket = transport

    def capture_and_render(event: ThinkEvent) -> None:
        events.append(event)
        renderer(event)

    client = make_client(
        auth_state,
        transport,
        on_event=capture_and_render,
    )
    run = client.submit("Research")

    socket.trigger(
        "new_message",
        {
            "id": "progress-card",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "🔍 Gathering the evidence…",
            "metadata": {
                "section": "progress",
                "progress": {
                    "done": False,
                    "phases": [
                        {
                            "id": "search-123",
                            "kind": "search",
                            "state": "active",
                            "label": "Gathering the evidence…",
                            "detail": "PubMed and FDA",
                            "count": {"done": 3, "total": 8},
                            "started_at": 1_786_752_000_000,
                        }
                    ],
                },
            },
        },
    )

    progress_events = [event for event in events if event.progress is not None]
    assert len(progress_events) == 1
    update = progress_events[0].progress
    assert isinstance(update, ProgressUpdate)
    assert update.done is False
    assert update.active_phase == ProgressPhase(
        id="search-123",
        kind="search",
        state="active",
        label="Gathering the evidence…",
        detail="PubMed and FDA",
        completed=3,
        total=8,
        started_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
        duration_seconds=None,
    )
    assert run.messages == ()
    assert "Gathering the evidence… — PubMed and FDA (3/8)" in output.getvalue()


def test_progress_renderer_emits_elapsed_heartbeat_without_server_frames() -> None:
    output = StringIO()
    renderer = ProgressRenderer(file=output, heartbeat_interval=0.02)
    now = datetime.now(timezone.utc)

    renderer(ThinkEvent(1, EventKind.SUBMITTED, "thread-1", now))
    deadline = time.monotonic() + 0.5
    while "Still working" not in output.getvalue() and time.monotonic() < deadline:
        time.sleep(0.01)
    renderer(
        ThinkEvent(2, EventKind.COMPLETED, "thread-1", datetime.now(timezone.utc))
    )
    rendered_after_completion = output.getvalue()
    time.sleep(0.05)

    assert "Still working" in rendered_after_completion
    assert output.getvalue() == rendered_after_completion


def test_detach_stops_the_run_progress_renderer(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    output = StringIO()
    renderer = ProgressRenderer(file=output, heartbeat_interval=0.02)
    client = make_client(auth_state, transport)
    run = client.submit_with_progress("Analyze", on_event=renderer)

    run.detach()
    rendered_after_detach = output.getvalue()
    time.sleep(0.05)

    assert output.getvalue() == rendered_after_detach


def test_detach_stops_the_canonical_show_progress_renderer(
    monkeypatch: pytest.MonkeyPatch,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    output = StringIO()
    renderer = ProgressRenderer(file=output, heartbeat_interval=0.02)
    monkeypatch.setattr(
        "bystro.think.progress._DEFAULT_PROGRESS_RENDERER",
        renderer,
    )
    client = make_client(
        auth_state,
        transport,
        on_event=think_client_module.show_progress,
    )
    run = client.submit("Analyze")

    run.detach()
    rendered_after_detach = output.getvalue()
    time.sleep(0.05)

    assert output.getvalue() == rendered_after_detach


def test_progress_renderer_does_not_reprint_the_unchanged_prefix_on_correction() -> None:
    output = StringIO()
    renderer = ProgressRenderer(file=output)
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    prefix = "stable-prefix-" * 20
    renderer(ThinkEvent(1, EventKind.SUBMITTED, "thread-1", now))
    renderer(
        ThinkEvent(
            2,
            EventKind.STREAM,
            "thread-1",
            now,
            stream_update=StreamUpdate(
                "answer",
                prefix + "old ending",
                "append",
                len(prefix) + len("old ending"),
                "assistant_message",
            ),
        )
    )
    renderer(
        ThinkEvent(
            3,
            EventKind.STREAM,
            "thread-1",
            now,
            stream_update=StreamUpdate(
                "answer",
                prefix + "new ending",
                "replace",
                len(prefix) + len("new ending"),
                "assistant_message",
            ),
        )
    )

    rendered = output.getvalue()
    assert rendered.count(prefix) == 1
    assert f"[stream updated from character {len(prefix)}]" in rendered
    assert rendered.endswith("new ending")


def test_progress_renderer_coalesces_replayed_needs_input_checkpoint(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit_with_progress("Analyze")
    before_commit = {
        "threadId": run.id,
        "visible": True,
        "waitingForInput": True,
        "humanApprovalStage": "clarifying_questions",
    }
    after_commit = {
        **before_commit,
        "checkpointId": "1f13d2a1-9893-603b-8000-000000000001",
    }

    socket.trigger("status_overlay_update", before_commit)
    socket.trigger("status_overlay_update", after_commit)

    run.respond("skip")
    socket.trigger(
        "status_overlay_update",
        {
            **before_commit,
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000002",
        },
    )

    output = capsys.readouterr().out
    assert output.count("[needs_input] Input required") == 2


def test_progress_renderer_coalesces_started_replay_around_input_response(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit_with_progress("Analyze")
    socket.trigger("task_start", {"threadId": run.id})
    socket.trigger(
        "new_message",
        {
            "id": "question",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Which cohort?",
            "metadata": {},
        },
    )
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": run.id,
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000001",
        },
    )

    def start_before_response_ack(
        event: str,
        data: object,
        timeout: float | None,
    ) -> None:
        del data, timeout
        if event == "client_message":
            socket.trigger("task_start", {"threadId": run.id})

    socket.call_hook = start_before_response_ack
    run.respond("Use all cohorts")
    socket.trigger("task_start", {"threadId": run.id})

    lines = capsys.readouterr().out.splitlines()
    assert lines.count("[submitted] Workload submitted") == 2
    assert lines.count("[started] Analysis started") == 2
    second_submission = lines.index("[submitted] Workload submitted", 2)
    assert lines[second_submission + 1] == "[started] Analysis started"


def test_websocket_stream_snapshots_become_incremental_typed_events(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    events: list[ThinkEvent] = []
    _session, socket = transport
    client = make_client(auth_state, transport, on_event=events.append)
    run = client.submit("Analyze")

    socket.trigger(
        "new_message",
        {
            "id": "answer",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Hel",
            "metadata": {"stream_type": "text"},
        },
    )
    socket.trigger(
        "stream_start",
        {
            "id": "answer",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Hel",
            "metadata": {"stream_type": "text"},
        },
    )
    socket.trigger(
        "stream_token",
        {
            "id": "answer",
            "threadId": run.id,
            "token": "Hello",
            "isSequence": True,
        },
    )
    socket.trigger(
        "stream_token",
        {
            "id": "answer",
            "threadId": run.id,
            "token": "Hello",
            "isSequence": True,
        },
    )
    socket.trigger(
        "stream_token",
        {
            "id": "answer",
            "threadId": run.id,
            "token": "!",
            "isSequence": False,
        },
    )
    socket.trigger(
        "update_message",
        {
            "id": "answer",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Hello! Done.",
            "metadata": {"stream_type": "text", "is_final_response": True},
        },
    )

    stream_updates = [
        event.stream_update
        for event in events
        if event.kind is EventKind.STREAM and event.stream_update is not None
    ]
    assert [update.delta for update in stream_updates] == [
        "Hel",
        "lo",
        "!",
        " Done.",
    ]
    assert [update.operation for update in stream_updates] == [
        "append",
        "append",
        "append",
        "append",
    ]
    assert [update.content_length for update in stream_updates] == [
        3,
        5,
        6,
        12,
    ]
    assert all(update.message_id == "answer" for update in stream_updates)


def test_streaming_progress_hides_reasoning_and_prints_visible_deltas(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit_with_progress("Analyze")

    socket.trigger(
        "stream_start",
        {
            "id": "reasoning",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "private reasoning text",
            "metadata": {"stream_type": "reasoning", "is_reasoning": True},
        },
    )
    socket.trigger(
        "stream_start",
        {
            "id": "answer",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Visible",
            "metadata": {"stream_type": "text", "is_reasoning": False},
        },
    )
    socket.trigger(
        "stream_token",
        {
            "id": "answer",
            "threadId": run.id,
            "token": "Visible answer",
            "isSequence": True,
        },
    )

    output = capsys.readouterr().out
    assert "private reasoning text" not in output
    assert output.count("[progress] Thinking...") == 1
    assert "[stream]\nVisible answer" in output


def test_streaming_progress_strips_terminal_control_sequences(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit_with_progress("Analyze")

    socket.trigger(
        "stream_start",
        {
            "id": "answer",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "safe\x1b[31m answer\x1b[0m",
            "metadata": {"stream_type": "text"},
        },
    )

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "safe[31m answer[0m" in output


def test_websocket_stream_events_redact_reasoning_content(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    events: list[ThinkEvent] = []
    _session, socket = transport
    client = make_client(auth_state, transport, on_event=events.append)
    run = client.submit("Analyze")

    socket.trigger(
        "stream_start",
        {
            "id": "reasoning",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "private reasoning text",
            "metadata": {"stream_type": "reasoning", "is_reasoning": True},
        },
    )
    socket.trigger(
        "stream_token",
        {
            "id": "reasoning",
            "threadId": run.id,
            "token": "private reasoning text continued",
            "isSequence": True,
        },
    )
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": run.id,
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
        },
    )

    stream_updates = [
        event.stream_update
        for event in events
        if event.kind is EventKind.STREAM and event.stream_update is not None
    ]
    assert len(stream_updates) == 1
    assert stream_updates[0].delta == ""
    assert stream_updates[0].is_reasoning is True
    assert stream_updates[0].content_length == len("private reasoning text")
    message_events = [event for event in events if event.kind is EventKind.MESSAGE]
    assert len(message_events) == 1
    assert message_events[0].message is None
    assert message_events[0].data["is_reasoning"] is True
    assert run.messages == ()
    assert run.needs_input is not None
    assert run.needs_input.prompt is None
    assert all("private reasoning text" not in repr(event) for event in events)


def test_websocket_stream_reports_replacements_and_retractions(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    events: list[ThinkEvent] = []
    _session, socket = transport
    client = make_client(auth_state, transport, on_event=events.append)
    run = client.submit("Analyze")

    socket.trigger(
        "stream_start",
        {
            "id": "answer",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Draft",
            "metadata": {"stream_type": "text"},
        },
    )
    socket.trigger(
        "stream_token",
        {
            "id": "answer",
            "threadId": run.id,
            "token": "Corrected",
            "isSequence": True,
        },
    )
    socket.trigger("delete_message", {"id": "answer", "threadId": run.id})

    stream_updates = [
        event.stream_update
        for event in events
        if event.kind is EventKind.STREAM and event.stream_update is not None
    ]
    assert [(update.operation, update.delta) for update in stream_updates] == [
        ("append", "Draft"),
        ("replace", "Corrected"),
        ("retract", ""),
    ]


@settings(max_examples=30, deadline=None, database=None)
@given(
    fragments=st.lists(
        st.text(alphabet=" abcdefghijklmnopqrstuvwxyz", min_size=0, max_size=12),
        min_size=1,
        max_size=20,
    )
)
def test_websocket_cumulative_snapshots_have_linear_incremental_payloads(
    fragments: list[str],
) -> None:
    auth_state = CachedAuth(
        email="scientist@example.com",
        access_token="dashboard-jwt",
        url="https://bystro.cloud",
    )
    transport = (FakeSession(), FakeSocket())
    _session, socket = transport
    events: list[ThinkEvent] = []
    client = make_client(auth_state, transport, on_event=events.append)
    run = client.submit("Analyze")
    cumulative = fragments[0]

    socket.trigger(
        "stream_start",
        {
            "id": "answer",
            "threadId": run.id,
            "type": "assistant_message",
            "output": cumulative,
            "metadata": {"stream_type": "text"},
        },
    )
    for fragment in fragments[1:]:
        cumulative += fragment
        socket.trigger(
            "stream_token",
            {
                "id": "answer",
                "threadId": run.id,
                "token": cumulative,
                "isSequence": True,
            },
        )

    stream_updates = [
        event.stream_update
        for event in events
        if event.kind is EventKind.STREAM and event.stream_update is not None
    ]
    assert all(update.operation == "append" for update in stream_updates)
    assert "".join(update.delta for update in stream_updates) == cumulative
    assert sum(len(update.delta) for update in stream_updates) == len(cumulative)


def test_output_files_and_downloads_use_authenticated_streaming_routes(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    row = {
        "name": "duck final.png",
        "path": "images/duck final.png",
        "size": 7,
        "modified": 2,
        "created": 1,
    }
    session.output_responses.extend(
        [
            FakeResponse(
                200,
                {"files": [row], "hasMore": True},
            ),
            FakeResponse(200, {"files": [], "hasMore": False}),
            FakeResponse(200, None, chunks=(b"PNG", b"DATA")),
            FakeResponse(200, None, chunks=(b"TAR", b"DATA")),
        ]
    )
    client = make_client(auth_state, transport)
    run = client.submit("Draw a duck")

    files = run.output_files(page_size=1)
    image = run.download_file(files[0], tmp_path / "duck.png", chunk_size=3)
    archive = run.download_all(tmp_path, chunk_size=4)

    assert files == (OutputFile("duck final.png", "images/duck final.png", 7, 2.0, 1.0),)
    assert image.read_bytes() == b"PNGDATA"
    assert archive.read_bytes() == b"TARDATA"
    output_gets = session.gets[-4:]
    assert [
        cast(dict[str, object], options["params"])["offset"]
        for _, options in output_gets[:2]
    ] == [0, 1]
    assert output_gets[2][0].endswith("/api/user-output/download/images/duck%20final.png")
    assert output_gets[2][1]["stream"] is True
    gets_before = len(session.gets)
    with pytest.raises(ValueError):
        run.download_file("../secret", tmp_path / "unsafe")
    assert len(session.gets) == gets_before


def test_output_download_allows_doubled_dots_inside_a_file_name(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    session.output_responses.append(FakeResponse(200, None, chunks=(b"RESULT",)))
    client = make_client(auth_state, transport)
    run = client.submit("Create a report")

    downloaded = run.download_file(
        "reports/report..final.txt",
        tmp_path / "report..final.txt",
    )

    assert downloaded.read_bytes() == b"RESULT"
    assert session.gets[-1][0].endswith(
        "/api/user-output/download/reports/report..final.txt"
    )


def test_list_conversations_searches_and_paginates(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    session.conversation_responses.extend(
        [
            FakeResponse(
                200,
                {
                    "data": [
                        {
                            "id": "thread-2",
                            "name": "CAR-T update",
                            "createdAt": "2026-08-15T18:24:30.123Z",
                        }
                    ],
                    "pageInfo": {"hasNextPage": True, "endCursor": "thread-2"},
                },
            ),
            FakeResponse(
                200,
                {
                    "data": [{"id": "thread-1", "name": None}],
                    "pageInfo": {"hasNextPage": False, "endCursor": "thread-1"},
                },
            ),
        ]
    )
    client = make_client(auth_state, transport)

    conversations = client.list_conversations(search=" CAR-T ", page_size=1)
    assert conversations[0] == PreviousConversation(
        "thread-2",
        "CAR-T update",
        datetime(2026, 8, 15, 18, 24, 30, 123000, tzinfo=timezone.utc),
    )
    assert conversations[1] == PreviousConversation("thread-1")
    requests = [options for url, options in session.posts if url.endswith("/project/threads")]
    assert requests[0]["json"] == {
        "pagination": {"first": 1, "cursor": None},
        "filter": {"search": "CAR-T"},
    }
    assert requests[1]["json"] == {
        "pagination": {"first": 1, "cursor": "thread-2"},
        "filter": {"search": "CAR-T"},
    }


def test_list_conversations_can_bound_history_without_fetching_every_page(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    session.conversation_responses.extend(
        [
            FakeResponse(
                200,
                {
                    "data": [{"id": "thread-new", "name": "Newest"}],
                    "pageInfo": {"hasNextPage": True, "endCursor": "thread-new"},
                },
            ),
            FakeResponse(
                200,
                {
                    "data": [{"id": "thread-old", "name": "Older"}],
                    "pageInfo": {"hasNextPage": False, "endCursor": "thread-old"},
                },
            ),
        ]
    )
    client = make_client(auth_state, transport)

    conversations = client.list_conversations(page_size=50, limit=1)

    assert conversations == (PreviousConversation("thread-new", "Newest"),)
    requests = [options for url, options in session.posts if url.endswith("/project/threads")]
    assert len(requests) == 1
    assert requests[0]["json"] == {
        "pagination": {"first": 1, "cursor": None},
        "filter": {"search": None},
    }


def test_list_conversations_rejects_nonpositive_limit(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    client = make_client(auth_state, transport)

    with pytest.raises(ValueError, match="limit"):
        client.list_conversations(limit=0)


def test_non_json_unauthorized_response_still_has_typed_auth_error(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    session.artifact_responses["private"] = InvalidJSONResponse(401, "login")
    client = make_client(auth_state, transport)

    with pytest.raises(ThinkAuthenticationError) as raised:
        client.get_artifact("private")

    assert raised.value.status_code == 401
    assert session.gets[0][1]["allow_redirects"] is False


def test_chunked_upload_is_bounded_retryable_and_reports_progress(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    source = tmp_path / "cohort.vcf"
    source.write_bytes(b"abcdefghij")
    session.upload_responses.extend(
        [
            FakeResponse(503, {"detail": "temporary"}),
            FakeResponse(200, {"completed": False}),
            FakeResponse(
                200,
                {
                    "completed": True,
                    "file": {
                        "id": "file-1",
                        "name": "cohort.vcf",
                        "displayName": "inputs/cohort.vcf",
                        "size": 10,
                        "mime": "text/vcf",
                    },
                },
            ),
        ]
    )
    sleeps: list[float] = []
    progress: list[UploadProgress] = []
    client = make_client(
        auth_state,
        transport,
        upload_chunk_size=5,
        sleep=sleeps.append,
    )

    uploaded = client.upload(
        source,
        artifact_path="inputs/cohort.vcf",
        on_progress=progress.append,
    )

    assert uploaded == UploadedFile(
        id="file-1",
        name="cohort.vcf",
        display_name="inputs/cohort.vcf",
        size=10,
        mime="text/vcf",
    )
    upload_posts = [entry for entry in session.posts if entry[0].endswith("/user/files/chunk")]
    assert len(upload_posts) == 3
    assert all(options["allow_redirects"] is False for _, options in upload_posts)
    first_data = upload_posts[0][1]["data"]
    assert isinstance(first_data, dict)
    assert first_data["chunkIndex"] == "0"
    assert first_data["totalChunks"] == "2"
    assert first_data["fileSize"] == "10"
    assert first_data["artifact_path"] == "inputs/cohort.vcf"
    assert first_data["chunkChecksum"] == (
        "36bbe50ed96841d10443bcb670d6554f0a34b761be67ec9c4a8ad2c0c44ca42c"
    )
    retry_data = upload_posts[1][1]["data"]
    assert isinstance(retry_data, dict)
    assert retry_data["uploadId"] == first_data["uploadId"]
    assert retry_data["chunkChecksum"] == first_data["chunkChecksum"]
    assert sleeps == [1.0]
    assert progress[-1].phase is UploadPhase.COMPLETE
    assert progress[-1].bytes_sent == 10
    assert progress[-1].total_bytes == 10


def test_upload_rejects_mismatched_artifact_path_before_connecting(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    source = tmp_path / "cohort.vcf"
    source.write_bytes(b"variant data")
    client = make_client(auth_state, transport)

    with pytest.raises(
        ValueError,
        match="Artifact path must end with the uploaded file name",
    ):
        client.upload(source, artifact_path="inputs/renamed.vcf")

    assert socket.connect_args is None
    assert session.posts == []


def test_chunked_upload_polls_async_finalization(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    source = tmp_path / "large.bam"
    source.write_bytes(b"abcdef")
    session.upload_responses.extend(
        [
            FakeResponse(200, {"completed": False}),
            FakeResponse(200, {"completed": False}),
        ]
    )
    session.status_responses.extend(
        [
            FakeResponse(200, {"status": "processing"}),
            FakeResponse(
                200,
                {
                    "status": "done",
                    "file": {
                        "id": "file-large",
                        "name": "large.bam",
                        "displayName": "large.bam",
                        "size": 6,
                        "mime": "application/octet-stream",
                    },
                },
            ),
        ]
    )
    sleeps: list[float] = []
    progress: list[UploadProgress] = []
    client = make_client(
        auth_state,
        transport,
        upload_chunk_size=3,
        sleep=sleeps.append,
    )

    uploaded = client.upload_artifact(source, on_progress=progress.append)

    assert uploaded.id == "file-large"
    assert [update.phase for update in progress][-2:] == [
        UploadPhase.FINALIZING,
        UploadPhase.COMPLETE,
    ]
    assert sleeps == [1.0]
    assert len(session.gets) == 2
    assert all("/user/files/chunk/status?uploadId=" in url for url, _ in session.gets)
    assert all(options["allow_redirects"] is False for _, options in session.gets)


def test_upload_finalization_fails_fast_on_nonretryable_http_error(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    source = tmp_path / "missing.vcf"
    source.write_bytes(b"content")
    session.upload_responses.append(FakeResponse(200, {"completed": False}))
    session.status_responses.append(FakeResponse(404, {"detail": "Upload not found"}))
    client = make_client(auth_state, transport)

    with pytest.raises(ThinkHTTPError) as raised:
        client.upload(source)

    assert raised.value.status_code == 404
    assert len(session.gets) == 1


def test_upload_progress_callback_failure_does_not_abort_artifact_creation(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
    caplog: pytest.LogCaptureFixture,
) -> None:
    session, _socket = transport
    source = tmp_path / "cohort.vcf"
    source.write_bytes(b"content")
    session.upload_responses.append(
        FakeResponse(
            200,
            {
                "completed": True,
                "file": {
                    "id": "file-callback",
                    "name": "cohort.vcf",
                    "displayName": "cohort.vcf",
                    "size": 7,
                    "mime": "text/vcf",
                },
            },
        )
    )
    client = make_client(auth_state, transport)

    def broken_callback(_progress: UploadProgress) -> None:
        raise RuntimeError("callback bug")

    uploaded = client.upload(source, on_progress=broken_callback)

    assert uploaded.id == "file-callback"
    assert "upload progress callback failed" in caplog.text.lower()


def test_submit_uploads_multiple_files_and_attaches_them_to_same_question(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    first = tmp_path / "variants.vcf"
    second = tmp_path / "phenotypes.tsv"
    first.write_bytes(b"vcf")
    second.write_bytes(b"tsv")
    session.upload_responses.extend(
        [
            FakeResponse(
                200,
                {
                    "completed": True,
                    "file": {
                        "id": "file-vcf",
                        "name": "variants.vcf",
                        "displayName": "variants.vcf",
                        "size": 3,
                        "mime": "text/vcf",
                    },
                },
            ),
            FakeResponse(
                200,
                {
                    "completed": True,
                    "file": {
                        "id": "file-tsv",
                        "name": "phenotypes.tsv",
                        "displayName": "phenotypes.tsv",
                        "size": 3,
                        "mime": "text/tab-separated-values",
                    },
                },
            ),
        ]
    )
    client = make_client(auth_state, transport)

    client.submit("Analyze these files together", files=[first, second])

    sent = [data for event, data, _timeout in socket.calls if event == "client_message"][-1]
    assert isinstance(sent, dict)
    message = sent["message"]
    assert isinstance(message, dict)
    assert message["output"] == "Analyze these files together"
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    attachments = metadata["attached_input_artifacts"]
    assert isinstance(attachments, list)
    assert [attachment["id"] for attachment in attachments] == ["file-vcf", "file-tsv"]


def test_submit_accepts_one_file_path_without_wrapping_it_in_a_list(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    source = tmp_path / "cohort.csv"
    source.write_bytes(b"sample,value\nA,1\n")
    session.upload_responses.append(
        FakeResponse(
            200,
            {
                "completed": True,
                "file": {
                    "id": "file-csv",
                    "name": "cohort.csv",
                    "displayName": "cohort.csv",
                    "size": source.stat().st_size,
                    "mime": "text/csv",
                },
            },
        )
    )
    client = make_client(auth_state, transport)

    client.submit("Analyze this file", files=str(source))

    sent = [data for event, data, _timeout in socket.calls if event == "client_message"][-1]
    assert isinstance(sent, dict)
    message = sent["message"]
    assert isinstance(message, dict)
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    attachments = metadata["attached_input_artifacts"]
    assert isinstance(attachments, list)
    assert [attachment["id"] for attachment in attachments] == ["file-csv"]


def test_submit_needs_input_respond_and_finish(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    uploaded = UploadedFile(
        id="file-1",
        name="cohort.vcf",
        display_name="Cohort",
        size=100,
        mime="text/vcf",
    )

    run = client.submit(
        "Analyze this cohort",
        files=[uploaded],
        datasets=[Dataset(id="job-1", name="Cases", assembly="hg38")],
        options=RunOptions(mode="plus", advanced_planning=True),
    )

    assert run.id == "thread-1"
    assert run.status is RunStatus.QUEUED
    send = next(data for event, data, _timeout in socket.calls if event == "client_message")
    assert isinstance(send, dict)
    assert send["new"] is True
    message = send["message"]
    assert isinstance(message, dict)
    assert message["output"] == "Analyze this cohort"
    assert message["metadata"] == {
        "mode": "plus",
        "advancedPlanningEnabled": True,
        "autoCompactEnabled": False,
        "fastEnabled": False,
        "verificationEnabled": True,
        "searchVerificationEnabled": True,
        "attached_input_artifacts": [
            {
                "id": "file-1",
                "name": "cohort.vcf",
                "displayName": "Cohort",
                "size": 100,
                "mime": "text/vcf",
                "scope": "personal",
            }
        ],
        "attached_bystro_datasets": [
            {"id": "job-1", "name": "Cases", "assembly": "hg38"}
        ],
    }

    socket.trigger("task_start", {"threadId": "thread-1", "taskId": "turn-1"})
    socket.trigger(
        "new_message",
        {
            "id": "question-1",
            "threadId": "thread-1",
            "type": "assistant_message",
            "output": "Which phenotype column should I use?",
            "metadata": {},
        },
    )
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": "thread-1",
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000030",
        },
    )

    paused = run.wait(timeout=0.1)
    assert paused == NeedsInput(
        run_id="thread-1",
        kind=InputKind.CLARIFICATION,
        prompt="Which phenotype column should I use?",
        checkpoint_id="1f13d2a1-9893-603b-8000-000000000030",
        details={},
    )

    socket.next_message_ack = {
        "success": True,
        "threadId": "thread-1",
        "created": False,
        "dispatched": True,
    }
    run.respond("Use case_control")
    assert run.status is RunStatus.QUEUED
    follow_up = [data for event, data, _timeout in socket.calls if event == "client_message"][-1]
    assert isinstance(follow_up, dict)
    assert follow_up["new"] is False
    follow_up_message = follow_up["message"]
    assert isinstance(follow_up_message, dict)
    assert follow_up_message["threadId"] == "thread-1"
    assert follow_up_message["metadata"] == {
        "mode": "plus",
        "advancedPlanningEnabled": True,
        "autoCompactEnabled": False,
        "fastEnabled": False,
        "isPlanningResponse": True,
    }

    # A replay of the checkpoint that was just answered cannot reopen the pause.
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": "thread-1",
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000030",
        },
    )
    assert run.status is RunStatus.QUEUED

    socket.trigger("task_start", {"threadId": "thread-1", "taskId": "turn-2"})
    socket.trigger(
        "new_message",
        {
            "id": "final-1",
            "threadId": "thread-1",
            "type": "assistant_message",
            "output": "The association analysis is complete.",
            "metadata": {"is_final_response": True},
        },
    )
    socket.trigger("task_end", {"threadId": "thread-1", "taskId": "turn-2"})

    result = run.wait(timeout=0.1)
    assert result == RunResult(
        run_id="thread-1",
        output="The association analysis is complete.",
    )
    assert run.status is RunStatus.SUCCEEDED


def test_respond_synchronizes_an_unstamped_pause_before_dispatch(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit(
        "Develop and review a plan",
        options=RunOptions(mode="plus", advanced_planning=True),
    )
    socket.trigger(
        "new_message",
        {
            "id": "question-unstamped",
            "threadId": "thread-1",
            "type": "assistant_message",
            "output": "Which phenotype should I use?",
            "metadata": {},
        },
    )
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": "thread-1",
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
        },
    )
    paused = run.wait(timeout=0.1)
    assert isinstance(paused, NeedsInput)
    assert paused.checkpoint_id is None

    def replay_stamped_pause(event: str, data: object) -> None:
        if event != "action" or not isinstance(data, dict):
            return
        if data.get("name") != "status_client_ready":
            return
        socket.trigger(
            "status_overlay_update",
            {
                "threadId": "thread-1",
                "visible": True,
                "waitingForInput": True,
                "humanApprovalStage": "clarifying_questions",
                "checkpointId": "1f13d2a1-9893-603b-8000-000000000040",
            },
        )

    socket.emit_hook = replay_stamped_pause
    socket.next_message_ack = {
        "success": True,
        "threadId": "thread-1",
        "created": False,
        "dispatched": True,
    }
    run.respond("Use case_control")

    assert run.status is RunStatus.QUEUED
    assert len([call for call in socket.calls if call[0] == "client_message"]) == 2
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": "thread-1",
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000040",
        },
    )
    assert run.status is RunStatus.QUEUED


def test_respond_retries_checkpoint_replay_after_stale_processing_overlay(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport, finalization_timeout=1.0)
    run = client.submit(
        "Develop and review a plan",
        options=RunOptions(mode="plus", advanced_planning=True),
    )
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": "thread-1",
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "plan_review",
        },
    )
    paused = run.wait(timeout=0.1)
    assert isinstance(paused, NeedsInput)
    assert paused.checkpoint_id is None

    replay_count = 0

    def replay_pause_after_stale_overlay(event: str, data: object) -> None:
        nonlocal replay_count
        if event != "action" or not isinstance(data, dict):
            return
        if data.get("name") != "status_client_ready":
            return
        replay_count += 1
        if replay_count == 1:
            socket.trigger(
                "status_overlay_update",
                {
                    "threadId": "thread-1",
                    "visible": True,
                    "waitingForInput": False,
                    "status": "processing",
                },
            )
            return
        socket.trigger(
            "status_overlay_update",
            {
                "threadId": "thread-1",
                "visible": True,
                "waitingForInput": True,
                "humanApprovalStage": "plan_review",
                "checkpointId": "1f13d2a1-9893-603b-8000-000000000041",
            },
        )

    socket.emit_hook = replay_pause_after_stale_overlay
    socket.next_message_ack = {
        "success": True,
        "threadId": "thread-1",
        "created": False,
        "dispatched": True,
    }

    run.respond("accept")

    assert replay_count >= 2
    assert run.status is RunStatus.QUEUED
    assert len([call for call in socket.calls if call[0] == "client_message"]) == 2


def test_respond_fails_closed_when_pause_checkpoint_cannot_be_synchronized(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    client = make_client(auth_state, transport, finalization_timeout=0.001)
    _session, socket = transport
    run = client.submit("Develop and review a plan")
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": "thread-1",
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
        },
    )

    with pytest.raises(RunProtocolError, match="checkpoint"):
        run.respond("Use case_control")

    assert run.status is RunStatus.NEEDS_INPUT
    assert len([call for call in socket.calls if call[0] == "client_message"]) == 1


def test_hidden_overlay_clears_needs_input_without_an_invalid_transient_state(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    checkpoint_id = "1f13d2a1-9893-603b-8000-000000000031"
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": run.id,
            "visible": True,
            "waitingForInput": True,
            "checkpointId": checkpoint_id,
        },
    )
    assert run.status is RunStatus.NEEDS_INPUT

    socket.trigger(
        "status_overlay_update",
        {
            "threadId": run.id,
            "visible": False,
            "waitingForInput": False,
            "checkpointId": checkpoint_id,
        },
    )

    assert run.needs_input is None
    assert run.status is RunStatus.RUNNING


def test_stream_token_keeps_final_output_current_after_task_end(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    socket.trigger("task_start", {"threadId": run.id})
    socket.trigger(
        "stream_start",
        {
            "id": "streamed-final",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "",
            "metadata": {"is_final_response": True},
        },
    )
    socket.trigger("task_end", {"threadId": run.id})
    socket.trigger(
        "stream_token",
        {
            "id": "streamed-final",
            "threadId": run.id,
            "token": "Durable answer",
            "isSequence": True,
        },
    )

    result = run.wait(timeout=0.1)

    assert isinstance(result, RunResult)
    assert result.output == "Durable answer"


def test_message_context_resolves_existing_artifact_and_uses_structured_metadata(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    session.artifact_responses["existing-file"] = FakeResponse(
        200,
        {
            "id": "existing-file",
            "name": "phenotypes.tsv",
            "displayName": "Phenotypes",
            "size": 200,
            "mime": "text/tab-separated-values",
        },
    )
    client = make_client(auth_state, transport)
    message: MessageWithContext = add_genetic_context("job-1", "Analyze", name="Cohort")
    message = add_previous_conversation_context("thread-prior", message, name="Prior")
    message = add_artifact_context("existing-file", message)

    client.submit(message)

    sent = [data for event, data, _timeout in socket.calls if event == "client_message"][-1]
    assert isinstance(sent, dict)
    payload = sent["message"]
    assert isinstance(payload, dict)
    assert payload["output"] == "Analyze"
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["attached_bystro_datasets"] == [
        {"id": "job-1", "name": "Cohort"}
    ]
    assert metadata["context_conversations"] == [
        {"id": "thread-prior", "name": "Prior"}
    ]
    assert metadata["attached_input_artifacts"] == [
        {
            "id": "existing-file",
            "name": "phenotypes.tsv",
            "displayName": "Phenotypes",
            "size": 200,
            "mime": "text/tab-separated-values",
            "scope": "personal",
        }
    ]


def test_submit_rejects_a_pre_ack_event_for_a_different_thread(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)

    def inject_stale_event(event: str, data: object, timeout: float | None) -> None:
        del data, timeout
        if event == "client_message":
            socket.trigger("task_start", {"threadId": "stale-thread"})

    socket.call_hook = inject_stale_event

    with pytest.raises(RunProtocolError, match="different run"):
        client.submit("Analyze")


def test_submit_preserves_a_matching_pre_ack_event(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)

    def inject_matching_event(event: str, data: object, timeout: float | None) -> None:
        del data, timeout
        if event == "client_message":
            socket.trigger("task_start", {"threadId": "thread-1"})

    socket.call_hook = inject_matching_event

    run = client.submit("Analyze")

    assert run.id == "thread-1"
    assert run.status is RunStatus.RUNNING
    assert any(
        event.kind is EventKind.STARTED and event.run_id == "thread-1"
        for event in run.history
    )


def test_threadless_task_end_cannot_complete_the_active_run(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    socket.trigger("task_start", {"threadId": run.id})
    socket.trigger("task_end", {})
    socket.trigger(
        "new_message",
        {
            "id": "final-after-threadless-end",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Done",
            "metadata": {"is_final_response": True},
        },
    )

    assert run.status is RunStatus.RUNNING


def test_threadless_resume_error_is_ignored_outside_transcript_hydration(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")

    socket.trigger("resume_thread_error", {"error": "stale resume failure"})

    assert run.status is RunStatus.QUEUED


def test_threadless_resume_error_fails_active_transcript_hydration(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.resume("thread-existing")

    socket.trigger("resume_thread_error", {"error": "resume denied"})

    with pytest.raises(RunProtocolError, match="resume denied"):
        run.wait(timeout=0.1)
    assert run.status is RunStatus.FAILED


def test_resume_hydrates_a_completed_run(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)

    run = client.resume("thread-existing")

    assert socket.connect_args is not None
    auth_callback = socket.connect_args[1]["auth"]
    assert callable(auth_callback)
    auth_payload = auth_callback()
    assert auth_payload["threadId"] == "thread-existing"
    assert any(
        event == "action" and isinstance(data, dict) and data.get("name") == "status_client_ready"
        for event, data in socket.emits
    )

    socket.trigger(
        "resume_thread",
        {
            "id": "thread-existing",
            "steps": [
                {
                    "id": "final-existing",
                    "threadId": "thread-existing",
                    "type": "assistant_message",
                    "output": "Persisted answer",
                    "metadata": {"is_final_response": True},
                }
            ],
            "elements": [],
        },
    )

    assert run.wait(timeout=0.1) == RunResult(
        run_id="thread-existing",
        output="Persisted answer",
    )


def test_resume_restores_run_options_for_follow_up_turns(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.resume("thread-mode")
    socket.trigger(
        "resume_thread",
        {
            "id": "thread-mode",
            "steps": [
                {
                    "id": "user-existing",
                    "threadId": "thread-mode",
                    "type": "user_message",
                    "output": "Deep analysis",
                    "metadata": {
                        "mode": "plus2",
                        "advancedPlanningEnabled": True,
                        "autoCompactEnabled": True,
                        "fastEnabled": True,
                        "verificationEnabled": False,
                        "searchVerificationEnabled": False,
                        "zdrEnabled": True,
                    },
                },
                {
                    "id": "final-existing",
                    "threadId": "thread-mode",
                    "type": "assistant_message",
                    "output": "Persisted answer",
                    "metadata": {"is_final_response": True},
                },
            ],
            "elements": [],
        },
    )
    assert isinstance(run.wait(timeout=0.1), RunResult)
    socket.next_message_ack = {
        "success": True,
        "threadId": "thread-mode",
        "created": True,
        "dispatched": True,
    }

    run.follow_up("Continue in the same mode")

    sent = [data for event, data, _timeout in socket.calls if event == "client_message"][-1]
    assert isinstance(sent, dict)
    message = sent["message"]
    assert isinstance(message, dict)
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["mode"] == "plus2"
    assert metadata["advancedPlanningEnabled"] is True
    assert metadata["autoCompactEnabled"] is True
    assert metadata["fastEnabled"] is True
    assert metadata["verificationEnabled"] is False
    assert metadata["searchVerificationEnabled"] is False
    assert metadata["zdrEnabled"] is True


def test_failed_resume_connection_does_not_poison_the_client(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    socket.connect_failure = OSError("offline")

    with pytest.raises(ThinkConnectionError, match="live connection"):
        client.resume("thread-unreachable")

    socket.connect_failure = None
    run = client.submit("A new workload")
    assert run.id == "thread-1"


def test_stale_run_cannot_refresh_the_socket_away_from_the_active_run(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    first = client.submit("First workload")
    socket.trigger("task_end", {"threadId": first.id})
    socket.trigger(
        "new_message",
        {
            "id": "first-final",
            "threadId": first.id,
            "type": "assistant_message",
            "output": "First answer",
            "metadata": {"is_final_response": True},
        },
    )
    second = client.submit("Second workload")
    calls_before = len(socket.calls)

    with pytest.raises(InputResponseError, match="no longer active"):
        first.refresh()

    assert len(socket.calls) == calls_before
    assert second.status is RunStatus.QUEUED


def test_resume_waits_for_transcript_before_returning_pause_prompt(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.resume("thread-paused")

    socket.trigger(
        "status_overlay_update",
        {
            "threadId": "thread-paused",
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000040",
        },
    )

    with pytest.raises(RunTimeoutError):
        run.wait(timeout=0.001)

    socket.trigger(
        "resume_thread",
        {
            "id": "thread-paused",
            "steps": [
                {
                    "id": "user-current",
                    "threadId": "thread-paused",
                    "type": "user_message",
                    "output": "Analyze the cohort",
                    "metadata": {},
                },
                {
                    "id": "question-current",
                    "threadId": "thread-paused",
                    "type": "assistant_message",
                    "output": "Which phenotype column should I use?",
                    "metadata": {},
                },
            ],
            "elements": [],
        },
    )

    assert run.wait(timeout=0.1) == NeedsInput(
        run_id="thread-paused",
        kind=InputKind.CLARIFICATION,
        prompt="Which phenotype column should I use?",
        checkpoint_id="1f13d2a1-9893-603b-8000-000000000040",
        details={},
    )


def test_resume_does_not_complete_new_turn_from_an_older_final_response(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.resume("thread-existing")

    socket.trigger(
        "resume_thread",
        {
            "id": "thread-existing",
            "steps": [
                {
                    "id": "user-old",
                    "threadId": "thread-existing",
                    "type": "user_message",
                    "output": "First question",
                    "metadata": {},
                },
                {
                    "id": "final-old",
                    "threadId": "thread-existing",
                    "type": "assistant_message",
                    "output": "First answer",
                    "metadata": {"is_final_response": True},
                },
                {
                    "id": "user-current",
                    "threadId": "thread-existing",
                    "type": "user_message",
                    "output": "Current follow-up",
                    "metadata": {},
                },
            ],
            "elements": [],
        },
    )

    assert run.status is RunStatus.QUEUED
    assert all(event.kind is not EventKind.COMPLETED for event in run.history)


def test_late_final_message_emits_completion_after_task_end(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")

    socket.trigger("task_start", {"threadId": run.id})
    socket.trigger("task_end", {"threadId": run.id})
    socket.trigger(
        "new_message",
        {
            "id": "final-late",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Late durable answer",
            "metadata": {"is_final_response": True},
        },
    )

    result = run.wait(timeout=0.1)
    assert isinstance(result, RunResult)
    assert result.output == "Late durable answer"
    assert [event.kind for event in run.history].count(EventKind.COMPLETED) == 1


def test_events_fail_closed_after_task_end_without_a_final_response(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport, finalization_timeout=0.001)
    run = client.submit("Analyze")
    socket.trigger("task_end", {"threadId": run.id})
    events = run.events(timeout=0.02)

    assert [next(events).kind, next(events).kind] == [
        EventKind.CONNECTED,
        EventKind.SUBMITTED,
    ]
    with pytest.raises(RunProtocolError, match="ended the task"):
        next(events)

    assert run.status is RunStatus.FAILED


def test_bounded_history_does_not_stall_event_iterator_after_eviction(
    monkeypatch,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    monkeypatch.setattr(think_client_module, "_MAX_EVENT_HISTORY", 3)
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    stream = run.events(timeout=0.1)

    assert next(stream).kind is EventKind.CONNECTED
    assert next(stream).kind is EventKind.SUBMITTED
    socket.trigger("task_start", {"threadId": run.id})
    assert next(stream).kind is EventKind.STARTED
    socket.trigger(
        "status_overlay_update",
        {"threadId": run.id, "visible": True, "display": "after eviction"},
    )

    assert next(stream).message == "after eviction"


def test_bounded_history_does_not_stall_wait_callback_after_eviction(
    monkeypatch,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    monkeypatch.setattr(think_client_module, "_MAX_EVENT_HISTORY", 3)
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    socket.trigger("task_start", {"threadId": run.id})
    observed: list[str] = []

    def on_event(event: ThinkEvent) -> None:
        if event.kind is EventKind.STARTED:
            socket.trigger(
                "status_overlay_update",
                {"threadId": run.id, "visible": True, "display": "after eviction"},
            )
        if event.message == "after eviction":
            observed.append(event.message)
            socket.trigger(
                "new_message",
                {
                    "id": "final-after-eviction",
                    "threadId": run.id,
                    "type": "assistant_message",
                    "output": "Done",
                    "metadata": {"is_final_response": True},
                },
            )
            socket.trigger("task_end", {"threadId": run.id})

    result = run.wait(timeout=0.2, on_event=on_event)

    assert isinstance(result, RunResult)
    assert result.output == "Done"
    assert observed == ["after eviction"]


def test_invalid_response_performs_no_file_or_artifact_io(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, _socket = transport
    source = tmp_path / "should-not-upload.txt"
    source.write_text("private")
    client = make_client(auth_state, transport)
    run = client.submit("Still running")
    posts_before = list(session.posts)

    with pytest.raises(InputResponseError, match="not waiting for input"):
        run.respond(
            add_artifact_context("should-not-fetch", "Too early"),
            files=[source],
        )

    assert session.posts == posts_before
    assert session.gets == []


def test_billing_pause_is_durable_but_cannot_be_answered_as_chat(
    tmp_path: Path,
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    source = tmp_path / "should-not-upload.txt"
    source.write_text("private")
    client = make_client(auth_state, transport)
    run = client.submit("Expensive analysis")
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": run.id,
            "visible": True,
            "waitingForInput": True,
            "status": "mid_turn_billing_required",
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000040",
            "billingRequired": {"cost": 3.25, "currency": "USD"},
        },
    )

    outcome = run.wait(timeout=0.1)

    assert isinstance(outcome, NeedsInput)
    assert outcome.kind is InputKind.BILLING
    assert outcome.details == {"cost": 3.25, "currency": "USD"}
    posts_before = list(session.posts)
    with pytest.raises(InputResponseError, match="paused for billing"):
        run.respond("continue", files=[source])
    assert session.posts == posts_before


def test_rejected_follow_up_restores_completed_turn(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("First question")
    socket.trigger(
        "new_message",
        {
            "id": "first-final",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "First answer",
            "metadata": {"is_final_response": True},
        },
    )
    socket.trigger("task_end", {"threadId": run.id})
    first_result = run.wait(timeout=0.1)
    assert isinstance(first_result, RunResult)
    assert first_result.output == "First answer"
    socket.next_message_ack = {
        "success": False,
        "error": "admission_rejected",
        "retryable": False,
    }

    with pytest.raises(RunRejectedError, match="admission_rejected"):
        run.follow_up("Second question")

    assert run.status is RunStatus.SUCCEEDED
    restored_result = run.wait(timeout=0.1)
    assert isinstance(restored_result, RunResult)
    assert restored_result.output == "First answer"


def test_run_result_lazily_exposes_structured_files_and_notebook_markdown(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Draw a duck")
    socket.trigger(
        "new_message",
        {
            "id": "final",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "## Duck\n\n![Duck](/api/user-files/thread-1/images/duck.png)",
            "metadata": {"is_final_response": True},
        },
    )
    socket.trigger("task_end", {"threadId": run.id})

    result = run.wait(timeout=0.1)

    assert isinstance(result, RunResult)
    assert session.gets == []
    session.output_responses.append(
        FakeResponse(
            200,
            {
                "files": [
                    {
                        "name": "duck.png",
                        "path": "images/duck.png",
                        "size": 1234,
                        "modified": 10.0,
                        "created": 9.0,
                    }
                ],
                "hasMore": False,
            },
        )
    )
    assert result.files == (
        OutputFile("duck.png", "images/duck.png", 1234, 10.0, 9.0),
    )
    assert result.artifacts is result.files
    assert len(session.gets) == 1
    assert result._repr_markdown_() == result.output
    assert hash(result) == hash(RunResult(result.run_id, result.output))


def test_interact_handles_clarification_callback_until_result(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    socket.trigger(
        "new_message",
        {
            "id": "question",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Which cohort?",
            "metadata": {},
        },
    )
    socket.trigger(
        "status_overlay_update",
        {
            "threadId": run.id,
            "visible": True,
            "waitingForInput": True,
            "humanApprovalStage": "clarifying_questions",
            "checkpointId": "1f13d2a1-9893-603b-8000-000000000001",
        },
    )

    def finish_after_response(
        event: str,
        data: object,
        timeout: float | None,
    ) -> None:
        del data, timeout
        if event != "client_message":
            return
        socket.trigger("task_start", {"threadId": run.id})
        socket.trigger(
            "new_message",
            {
                "id": "final",
                "threadId": run.id,
                "type": "assistant_message",
                "output": "All cohorts analyzed",
                "metadata": {"is_final_response": True},
            },
        )
        socket.trigger("task_end", {"threadId": run.id})

    socket.call_hook = finish_after_response
    seen: list[NeedsInput] = []

    def answer_clarification(request: NeedsInput) -> str:
        seen.append(request)
        return "all cohorts"

    result = run.interact(
        timeout=1.0,
        on_clarification=answer_clarification,
    )

    assert isinstance(result, RunResult)
    assert result.output == "All cohorts analyzed"
    assert [request.prompt for request in seen] == ["Which cohort?"]


def test_cancel_waits_for_durable_stop_release_and_detach_does_not_cancel(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    events: list[ThinkEvent] = []
    client = make_client(auth_state, transport, on_event=events.append)
    run = client.submit("Long analysis")
    socket.trigger("task_start", {"threadId": run.id, "taskId": "task-1"})

    def release_stop(event: str, data: object) -> None:
        if event != "stop":
            return
        assert data == {
            "threadId": run.id,
            "clientObservedActive": True,
            "taskId": "task-1",
        }
        socket.trigger("task_stopping", {"threadId": run.id, "taskId": "task-1"})
        socket.trigger("task_end", {"threadId": run.id, "taskId": "task-1"})
        socket.trigger(
            "thread_stop_released",
            {"threadId": run.id, "taskId": "task-1"},
        )

    socket.emit_hook = release_stop
    run.cancel(timeout=0.5)

    assert run.status is RunStatus.CANCELLED
    assert EventKind.CANCELLING in {event.kind for event in events}
    assert EventKind.CANCELLED in {event.kind for event in events}
    with pytest.raises(RunCancelledError):
        run.wait(timeout=0.1)

    second_transport = (FakeSession(), FakeSocket())
    _second_session, second_socket = second_transport
    second_client = make_client(auth_state, second_transport)
    second_run = second_client.submit("Keep running")
    second_socket.trigger("task_start", {"threadId": second_run.id})
    second_run.detach()
    assert second_socket.connected is False
    assert not any(event == "stop" for event, _data in second_socket.emits)


def test_cancel_is_reissued_after_reconnect_until_stop_release(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Long analysis")
    socket.trigger("task_start", {"threadId": run.id})

    with pytest.raises(RunTimeoutError):
        run.cancel(timeout=0.02)
    assert sum(event == "stop" for event, _data in socket.emits) == 1

    socket.connected = False
    client.connect()
    assert sum(event == "stop" for event, _data in socket.emits) == 2
    socket.trigger("thread_stop_released", {"threadId": run.id})
    assert run.status is RunStatus.CANCELLED
    socket.trigger("task_start", {"threadId": run.id, "taskId": "late-task"})
    socket.trigger("task_stopping", {"threadId": run.id, "taskId": "late-task"})
    assert run.status is RunStatus.CANCELLED


def test_stale_task_lifecycle_events_do_not_settle_the_active_turn(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    socket.trigger("task_start", {"threadId": run.id, "taskId": "current-task"})

    socket.trigger("task_end", {"threadId": run.id, "taskId": "old-task"})
    socket.trigger("task_stopping", {"threadId": run.id, "taskId": "old-task"})
    socket.trigger(
        "thread_stop_released",
        {"threadId": run.id, "taskId": "old-task"},
    )
    socket.trigger(
        "thread_stop_released",
        {"threadId": run.id, "taskId": "current-task"},
    )

    assert run.status is RunStatus.RUNNING
    with pytest.raises(RunTimeoutError):
        run.wait(timeout=0.01)


def test_follow_up_cancel_before_task_start_does_not_reuse_the_prior_task_id(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("First turn")
    socket.trigger("task_start", {"threadId": run.id, "taskId": "first-task"})
    socket.trigger(
        "new_message",
        {
            "id": "first-final",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "First answer",
            "metadata": {"is_final_response": True},
        },
    )
    socket.trigger("task_end", {"threadId": run.id, "taskId": "first-task"})
    assert isinstance(run.wait(timeout=0.1), RunResult)

    run.follow_up("Second turn")
    with pytest.raises(RunTimeoutError):
        run.cancel(timeout=0.01)

    stop_payloads = [data for event, data in socket.emits if event == "stop"]
    assert stop_payloads[-1] == {
        "threadId": run.id,
        "clientObservedActive": True,
    }
    socket.trigger("task_start", {"threadId": run.id, "taskId": "second-task"})
    assert run.status is RunStatus.CANCELLING
    assert socket.emits[-1] == (
        "stop",
        {
            "threadId": run.id,
            "clientObservedActive": True,
            "taskId": "second-task",
        },
    )


@settings(max_examples=50, deadline=None, database=None)
@given(
    lifecycle_events=st.lists(
        st.tuples(
            st.sampled_from(
                ["task_end", "task_stopping", "thread_stop_released"]
            ),
            st.sampled_from([None, "current-task", "stale-task"]),
        ),
        max_size=30,
    )
)
def test_task_lifecycle_routing_fuzz_matches_the_active_task_only(
    lifecycle_events: list[tuple[str, str | None]],
) -> None:
    auth_state = CachedAuth(
        email="scientist@example.com",
        access_token="dashboard-jwt",
        url="https://bystro.cloud",
    )
    transport = (FakeSession(), FakeSocket())
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    socket.trigger("task_start", {"threadId": run.id, "taskId": "current-task"})
    expected_status = RunStatus.RUNNING
    cancel_requested = False

    for event, task_id in lifecycle_events:
        payload: dict[str, object] = {"threadId": run.id}
        if task_id is not None:
            payload["taskId"] = task_id
        task_matches = task_id in {None, "current-task"}
        if expected_status is not RunStatus.CANCELLED:
            if event == "task_stopping" and task_matches:
                cancel_requested = True
                expected_status = RunStatus.CANCELLING
            elif (
                event == "thread_stop_released"
                and task_matches
                and cancel_requested
            ):
                expected_status = RunStatus.CANCELLED
        socket.trigger(event, payload)
        assert run.status is expected_status


def test_async_event_iterator_and_wait_return_the_terminal_result(
    auth_state: CachedAuth,
    transport: tuple[FakeSession, FakeSocket],
) -> None:
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    socket.trigger("task_start", {"threadId": run.id})
    socket.trigger(
        "new_message",
        {
            "id": "final",
            "threadId": run.id,
            "type": "assistant_message",
            "output": "Done",
            "metadata": {"is_final_response": True},
        },
    )
    socket.trigger("task_end", {"threadId": run.id})

    async def consume() -> tuple[list[ThinkEvent], RunResult]:
        events = [event async for event in run.aevents(timeout=1.0)]
        outcome = await run.await_result(timeout=1.0)
        assert isinstance(outcome, RunResult)
        return events, outcome

    events, result = asyncio.run(consume())
    assert EventKind.COMPLETED in {event.kind for event in events}
    assert result.output == "Done"


@settings(max_examples=50, deadline=None, database=None)
@given(
    message_kinds=st.lists(
        st.sampled_from(["user", "assistant", "final"]),
        min_size=1,
        max_size=20,
    )
)
def test_resume_transcript_fuzz_only_completes_the_latest_turn(
    message_kinds: list[str],
) -> None:
    auth_state = CachedAuth(
        email="scientist@example.com",
        access_token="dashboard-jwt",
        url="https://bystro.cloud",
    )
    transport = (FakeSession(), FakeSocket())
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.resume("thread-fuzz")
    steps: list[dict[str, object]] = []
    expected_output: str | None = None
    for index, message_kind in enumerate(message_kinds):
        output = f"output-{index}"
        if message_kind == "user":
            expected_output = None
            message_type = "user_message"
            metadata: dict[str, object] = {}
        else:
            message_type = "assistant_message"
            metadata = {"is_final_response": True} if message_kind == "final" else {}
            if message_kind == "final":
                expected_output = output
        steps.append(
            {
                "id": f"message-{index}",
                "threadId": "thread-fuzz",
                "type": message_type,
                "output": output,
                "metadata": metadata,
            }
        )

    socket.trigger(
        "resume_thread",
        {"id": "thread-fuzz", "steps": steps, "elements": []},
    )

    completed_events = [event for event in run.history if event.kind is EventKind.COMPLETED]
    if expected_output is None:
        assert run.status is RunStatus.QUEUED
        assert completed_events == []
    else:
        assert run.status is RunStatus.SUCCEEDED
        result = run.wait(timeout=0.1)
        assert isinstance(result, RunResult)
        assert result.output == expected_output
        assert len(completed_events) == 1


@settings(max_examples=30, deadline=None, database=None)
@given(
    checkpoint_numbers=st.lists(
        st.integers(min_value=1, max_value=999),
        min_size=1,
        max_size=25,
        unique=True,
    )
)
def test_needs_input_fuzz_converges_on_newest_checkpoint_and_rejects_replays(
    checkpoint_numbers: list[int],
) -> None:
    auth_state = CachedAuth(
        email="scientist@example.com",
        access_token="dashboard-jwt",
        url="https://bystro.cloud",
    )
    transport = (FakeSession(), FakeSocket())
    _session, socket = transport
    client = make_client(auth_state, transport)
    run = client.submit("Analyze")
    checkpoints = [
        f"1f13d2a1-9893-603b-8000-{number:012d}" for number in checkpoint_numbers
    ]
    for checkpoint_id in checkpoints:
        socket.trigger(
            "status_overlay_update",
            {
                "threadId": run.id,
                "visible": True,
                "waitingForInput": True,
                "humanApprovalStage": "clarifying_questions",
                "checkpointId": checkpoint_id,
            },
        )

    outcome = run.wait(timeout=0.1)

    assert isinstance(outcome, NeedsInput)
    assert outcome.checkpoint_id == max(checkpoints)
    run.respond("Use phenotype")
    for checkpoint_id in checkpoints:
        socket.trigger(
            "status_overlay_update",
            {
                "threadId": run.id,
                "visible": True,
                "waitingForInput": True,
                "humanApprovalStage": "clarifying_questions",
                "checkpointId": checkpoint_id,
            },
        )
    assert run.status is RunStatus.QUEUED
