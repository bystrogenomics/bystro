"""Terminal rendering for Bystro Think live events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import sys
import threading
import time
from typing import TextIO, TypeAlias

from bystro.think.models import EventKind, ProgressUpdate, StreamUpdate, ThinkEvent

_MAX_RENDER_STATES = 128
_GENERIC_PROGRESS_MESSAGES = frozenset({"Processing...", "Thinking..."})

ProgressCallback: TypeAlias = Callable[[ThinkEvent], None]

_PROGRESS_MESSAGES: dict[EventKind, str] = {
    EventKind.SUBMITTED: "Workload submitted",
    EventKind.STARTED: "Analysis started",
    EventKind.NEEDS_INPUT: "Input required",
    EventKind.CANCELLING: "Cancelling analysis",
    EventKind.CANCELLED: "Analysis cancelled",
    EventKind.COMPLETED: "Analysis complete",
}


def _terminal_text(value: str) -> str:
    """Remove terminal control bytes while preserving readable layout."""

    return "".join(
        character
        for character in value
        if character in {"\n", "\t"}
        or (ord(character) >= 32 and not 127 <= ord(character) <= 159)
    )


def _common_prefix_length(left: str, right: str) -> int:
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    return min(len(left), len(right))


@dataclass(slots=True)
class _ProgressRenderState:
    last_progress: str | None = None
    last_progress_at: datetime | None = None
    turn_started_at: datetime | None = None
    needs_input_rendered: bool = False
    stream_message_id: str | None = None
    visible_message_id: str | None = None
    visible_content: str = ""
    stream_line_open: bool = False
    turn: int = 0
    started_turn: int = -1
    pending_started: bool = False
    waiting_for_submission: bool = True
    turn_started_monotonic: float | None = None
    last_activity_monotonic: float | None = None
    generic_progress_seen: set[str] = field(default_factory=set)
    heartbeat_stop: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None


class ProgressRenderer:
    """Render coalesced lifecycle updates and live visible output."""

    def __init__(
        self,
        *,
        stream_output: bool = True,
        file: TextIO | None = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self._stream_output = stream_output
        self._file = file
        self._heartbeat_interval = heartbeat_interval
        self._states: dict[str, _ProgressRenderState] = {}
        self._lock = threading.RLock()

    def _output(self) -> TextIO:
        return self._file or sys.stdout

    def _state(self, run_id: str) -> _ProgressRenderState:
        state = self._states.get(run_id)
        if state is None:
            state = _ProgressRenderState()
            self._states[run_id] = state
            if len(self._states) > _MAX_RENDER_STATES:
                for oldest_run_id in tuple(self._states):
                    if oldest_run_id != run_id:
                        self._stop_heartbeat(self._states[oldest_run_id])
                        del self._states[oldest_run_id]
                        break
        return state

    def _close_stream_line(self, state: _ProgressRenderState) -> None:
        if state.stream_line_open:
            print(file=self._output(), flush=True)
            state.stream_line_open = False

    @staticmethod
    def _stop_heartbeat(state: _ProgressRenderState) -> None:
        stop = state.heartbeat_stop
        if stop is not None:
            stop.set()
        state.heartbeat_stop = None
        state.heartbeat_thread = None

    def _heartbeat_loop(
        self,
        run_id: str,
        turn: int,
        stop: threading.Event,
    ) -> None:
        while not stop.wait(self._heartbeat_interval):
            with self._lock:
                state = self._states.get(run_id)
                if (
                    state is None
                    or state.turn != turn
                    or state.heartbeat_stop is not stop
                    or state.waiting_for_submission
                ):
                    return
                now = time.monotonic()
                last_activity = state.last_activity_monotonic
                if (
                    last_activity is not None
                    and now - last_activity < self._heartbeat_interval
                ):
                    continue
                started_at = state.turn_started_monotonic or now
                elapsed = max(0, int(now - started_at))
                minutes, seconds = divmod(elapsed, 60)
                elapsed_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
                self._write_status(
                    state,
                    EventKind.PROGRESS,
                    f"Still working... ({elapsed_text} elapsed)",
                )
                state.last_progress_at = datetime.now(timezone.utc)
                state.last_activity_monotonic = now

    def _start_heartbeat(self, run_id: str, state: _ProgressRenderState) -> None:
        self._stop_heartbeat(state)
        stop = threading.Event()
        thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(run_id, state.turn, stop),
            name=f"bystro-think-progress-{run_id[:8]}",
            daemon=True,
        )
        state.heartbeat_stop = stop
        state.heartbeat_thread = thread
        thread.start()

    def _write_status(
        self,
        state: _ProgressRenderState,
        kind: EventKind,
        message: str,
    ) -> None:
        self._close_stream_line(state)
        state.stream_message_id = None
        print(
            f"[{kind.value}] {_terminal_text(message)}",
            file=self._output(),
            flush=True,
        )

    def _render_stream(
        self,
        state: _ProgressRenderState,
        update: StreamUpdate,
        created_at: datetime,
    ) -> None:
        state.last_activity_monotonic = time.monotonic()
        if update.is_reasoning:
            message = "Thinking..."
            if message in state.generic_progress_seen:
                return
            state.generic_progress_seen.add(message)
            self._write_status(state, EventKind.PROGRESS, message)
            state.last_progress = message
            state.last_progress_at = created_at
            return
        if not self._stream_output:
            return
        if update.operation == "retract":
            self._close_stream_line(state)
            print("[stream restarted]", file=self._output(), flush=True)
            state.stream_message_id = None
            if state.visible_message_id == update.message_id:
                state.visible_message_id = None
                state.visible_content = ""
            return
        same_visible_message = state.visible_message_id == update.message_id
        visible_delta = update.delta
        if update.operation == "replace" and same_visible_message:
            unchanged_prefix = _common_prefix_length(
                state.visible_content,
                update.delta,
            )
            self._close_stream_line(state)
            print(
                f"[stream updated from character {unchanged_prefix}]",
                file=self._output(),
                flush=True,
            )
            state.stream_message_id = update.message_id
            visible_delta = update.delta[unchanged_prefix:]
        elif state.stream_message_id != update.message_id:
            self._close_stream_line(state)
            print("[stream]", file=self._output(), flush=True)
            state.stream_message_id = update.message_id

        if update.operation == "replace":
            state.visible_content = update.delta
        elif same_visible_message:
            state.visible_content += update.delta
        else:
            state.visible_content = update.delta
        state.visible_message_id = update.message_id

        if visible_delta:
            terminal_delta = _terminal_text(visible_delta)
            output = self._output()
            output.write(terminal_delta)
            output.flush()
            state.stream_line_open = bool(terminal_delta) and not terminal_delta.endswith("\n")

    @staticmethod
    def _structured_progress_message(update: ProgressUpdate) -> str | None:
        phase = update.active_phase
        if phase is None:
            return None
        message = phase.label
        if phase.detail and phase.detail not in message:
            message = f"{message} — {phase.detail}"
        if (
            phase.completed is not None
            and phase.total is not None
            and phase.total > 0
        ):
            message = f"{message} ({phase.completed}/{phase.total})"
        return message

    def __call__(self, event: ThinkEvent) -> None:
        with self._lock:
            state = self._state(event.run_id)
            if event.kind is EventKind.STREAM:
                if event.stream_update is not None:
                    self._render_stream(state, event.stream_update, event.created_at)
                return
            if event.kind is EventKind.MESSAGE:
                return
            if event.kind is EventKind.NEEDS_INPUT:
                # The live overlay is first emitted before checkpoint commit and
                # replayed with its durable id. A response submission is the only
                # writer that can advance this run to another input pause.
                if state.needs_input_rendered:
                    return
                state.needs_input_rendered = True
            if (
                event.kind is EventKind.DISCONNECTED
                and event.message
                and "client disconnect" in event.message.casefold()
            ):
                return
            if event.kind is EventKind.SUBMITTED:
                self._stop_heartbeat(state)
                state.turn += 1
                state.started_turn = -1
                state.waiting_for_submission = False
                state.needs_input_rendered = False
                state.last_progress = None
                state.last_progress_at = None
                state.generic_progress_seen.clear()
                state.turn_started_at = event.created_at
                state.turn_started_monotonic = time.monotonic()
                state.last_activity_monotonic = state.turn_started_monotonic
                state.visible_message_id = None
                state.visible_content = ""
                self._write_status(state, event.kind, _PROGRESS_MESSAGES[event.kind])
                if state.pending_started:
                    self._write_status(
                        state,
                        EventKind.STARTED,
                        _PROGRESS_MESSAGES[EventKind.STARTED],
                    )
                    state.started_turn = state.turn
                    state.pending_started = False
                self._start_heartbeat(event.run_id, state)
                return
            if event.kind is EventKind.STARTED:
                if state.waiting_for_submission or state.turn == 0:
                    state.pending_started = True
                    return
                if state.started_turn == state.turn:
                    return
                state.started_turn = state.turn
                self._write_status(state, event.kind, _PROGRESS_MESSAGES[event.kind])
                return
            if event.kind is EventKind.PROGRESS:
                progress_message = (
                    self._structured_progress_message(event.progress)
                    if event.progress is not None
                    else None
                )
                if progress_message is None:
                    detail = event.data.get("detail")
                    progress_message = (
                        detail.strip()
                        if isinstance(detail, str) and detail.strip()
                        else event.message
                    )
                if progress_message is None or progress_message == "progress":
                    return
                generic_progress = (
                    event.progress is None
                    and progress_message in _GENERIC_PROGRESS_MESSAGES
                )
                repeated_generic = (
                    generic_progress
                    and progress_message in state.generic_progress_seen
                )
                if generic_progress:
                    state.generic_progress_seen.add(progress_message)
                if progress_message == state.last_progress or repeated_generic:
                    if state.last_progress_at is None:
                        state.last_progress_at = event.created_at
                        return
                    since_last = (
                        event.created_at - state.last_progress_at
                    ).total_seconds()
                    if since_last < self._heartbeat_interval:
                        return
                    state.last_progress_at = event.created_at
                    started_at = state.turn_started_at or event.created_at
                    elapsed = max(
                        0,
                        int((event.created_at - started_at).total_seconds()),
                    )
                    minutes, seconds = divmod(elapsed, 60)
                    elapsed_text = (
                        f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
                    )
                    message = f"Still working... ({elapsed_text} elapsed)"
                else:
                    state.last_progress = progress_message
                    state.last_progress_at = event.created_at
                    state.last_activity_monotonic = time.monotonic()
                    if state.turn_started_at is None:
                        state.turn_started_at = event.created_at
                    message = progress_message
            else:
                message = (
                    _PROGRESS_MESSAGES.get(event.kind, event.message)
                    or event.kind.value
                )
                state.last_activity_monotonic = time.monotonic()
            if event.kind in {
                EventKind.NEEDS_INPUT,
                EventKind.CANCELLED,
                EventKind.COMPLETED,
                EventKind.FAILED,
            }:
                state.waiting_for_submission = True
                self._stop_heartbeat(state)
            self._write_status(state, event.kind, message)
            if event.kind in {
                EventKind.CANCELLED,
                EventKind.COMPLETED,
                EventKind.FAILED,
            }:
                state.visible_message_id = None
                state.visible_content = ""

    def close(self, run_id: str | None = None) -> None:
        """Stop owned heartbeat workers, optionally for one run only."""

        with self._lock:
            if run_id is not None:
                state = self._states.pop(run_id, None)
                if state is not None:
                    self._stop_heartbeat(state)
                return
            for state in self._states.values():
                self._stop_heartbeat(state)
            self._states.clear()


_DEFAULT_PROGRESS_RENDERER = ProgressRenderer()


def show_progress(event: ThinkEvent) -> None:
    """Render coalesced progress and browser-visible streamed output."""

    _DEFAULT_PROGRESS_RENDERER(event)


def close_progress_callback(
    callback: ProgressCallback | None,
    run_id: str | None,
) -> None:
    """Release renderer state owned by a public progress callback."""

    if run_id is None:
        return
    if isinstance(callback, ProgressRenderer):
        callback.close(run_id)
    elif callback is show_progress:
        _DEFAULT_PROGRESS_RENDERER.close(run_id)


__all__ = ["ProgressCallback", "ProgressRenderer", "show_progress"]
