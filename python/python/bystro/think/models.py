"""Public value objects for the Bystro Think SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
from typing import Callable, Literal, Mapping, TypeAlias

ConversationMode: TypeAlias = Literal["base", "plus", "plus2", "phd"]
StreamOperation: TypeAlias = Literal["append", "replace", "retract"]


class RunStatus(str, Enum):
    """Lifecycle state of a Think run."""

    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InputKind(str, Enum):
    """The action required to resume a paused run."""

    CLARIFICATION = "clarification"
    PLAN_REVIEW = "plan_review"
    BILLING = "billing"


class EventKind(str, Enum):
    """Stable SDK event categories."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SUBMITTED = "submitted"
    STARTED = "started"
    PROGRESS = "progress"
    STREAM = "stream"
    MESSAGE = "message"
    NEEDS_INPUT = "needs_input"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadPhase(str, Enum):
    """Phase of a resumable upload."""

    UPLOADING = "uploading"
    FINALIZING = "finalizing"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class RunOptions:
    """The intentionally small set of supported Think run controls."""

    mode: ConversationMode = "base"
    advanced_planning: bool = False
    auto_compact: bool = False
    fast: bool = False
    verify: bool = True
    verify_sources: bool = True
    zero_data_retention: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"base", "plus", "plus2", "phd"}:
            raise ValueError(f"unsupported Think mode: {self.mode}")


@dataclass(frozen=True, slots=True)
class Dataset:
    """A Bystro dataset to attach to the agent context."""

    id: str  # noqa: A003 - mirrors the dataset API field
    name: str
    assembly: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("dataset id cannot be empty")
        if not self.name.strip():
            raise ValueError("dataset name cannot be empty")


@dataclass(frozen=True, slots=True)
class BillingTopUpApproval:
    """An explicit authorization to raise the fixed monthly spend cap."""

    amount_cents: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.amount_cents, bool)
            or not isinstance(self.amount_cents, int)
            or self.amount_cents <= 0
        ):
            raise ValueError("top-up amount must be a positive integer number of cents")


@dataclass(frozen=True, slots=True)
class BillingTopUpRequest:
    """The server-priced funding shortfall for one blocked message."""

    minimum_top_up_cents: int
    required_cost_cents: int
    current_monthly_limit_cents: int | None
    message: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def approve(self, amount_cents: int | None = None) -> BillingTopUpApproval:
        """Explicitly approve the minimum top-up, or a larger cent amount."""

        resolved_amount = self.minimum_top_up_cents if amount_cents is None else amount_cents
        approval = BillingTopUpApproval(resolved_amount)
        if approval.amount_cents < self.minimum_top_up_cents:
            raise ValueError(
                "approved top-up amount cannot be smaller than the required amount"
            )
        return approval


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """A durable personal input artifact returned by Think."""

    id: str  # noqa: A003 - mirrors the artifact API field
    name: str
    display_name: str
    size: int
    mime: str | None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("uploaded file id cannot be empty")
        if not self.name.strip():
            raise ValueError("uploaded file name cannot be empty")
        if not self.display_name.strip():
            raise ValueError("uploaded file display name cannot be empty")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("uploaded file size must be a non-negative integer")


FileInput: TypeAlias = UploadedFile | str | Path


@dataclass(frozen=True, slots=True)
class UploadProgress:
    """Progress update for a resumable file upload."""

    phase: UploadPhase
    bytes_sent: int
    total_bytes: int

    @property
    def fraction(self) -> float:
        """Return progress as a value from 0.0 through 1.0."""

        if self.total_bytes <= 0:
            return 1.0
        return min(1.0, self.bytes_sent / self.total_bytes)


@dataclass(frozen=True, slots=True)
class OutputFile:
    """A file generated in a Think conversation's output directory."""

    name: str
    path: str
    size: int
    modified: float
    created: float


@dataclass(frozen=True, slots=True)
class ThinkMessage:
    """A user or assistant message in the durable run transcript."""

    id: str  # noqa: A003 - mirrors the message API field
    run_id: str
    type: str  # noqa: A003 - mirrors the message API field
    output: str
    name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class StreamUpdate:
    """A typed update derived from a live Socket.IO output frame."""

    message_id: str
    delta: str
    operation: StreamOperation
    content_length: int
    message_type: str
    name: str | None = None
    stream_type: str | None = None
    is_reasoning: bool = False
    section: str | None = None


@dataclass(frozen=True, slots=True)
class ProgressPhase:
    """One backend-owned unit of work in a structured progress update."""

    id: str  # noqa: A003 - mirrors the progress protocol field
    kind: str
    state: str
    label: str
    detail: str | None = None
    completed: int | None = None
    total: int | None = None
    started_at: datetime | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """A complete snapshot of the server's current workload phases."""

    done: bool
    phases: tuple[ProgressPhase, ...]

    @property
    def active_phase(self) -> ProgressPhase | None:
        """Return the most relevant current phase, if the snapshot has one."""

        for state in ("active", "pending"):
            for phase in reversed(self.phases):
                if phase.state == state:
                    return phase
        return self.phases[-1] if self.phases else None


@dataclass(frozen=True, slots=True)
class ThinkEvent:
    """An ordered, SDK-level progress event."""

    sequence: int
    kind: EventKind
    run_id: str
    created_at: datetime
    message: str | None = None
    data: Mapping[str, object] = field(default_factory=dict)
    stream_update: StreamUpdate | None = None
    progress: ProgressUpdate | None = None


@dataclass(frozen=True, slots=True)
class NeedsInput:
    """A durable pause that must be resolved before work can continue."""

    run_id: str
    kind: InputKind
    prompt: str | None
    checkpoint_id: str | None
    details: Mapping[str, object] = field(default_factory=dict)

    def _repr_markdown_(self) -> str:
        """Render the durable input request naturally in notebooks."""

        return self.prompt or f"Input required: {self.kind.value}"


OutputFileLoader: TypeAlias = Callable[[], tuple[OutputFile, ...]]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Successful Think output with lazily discovered generated files."""

    run_id: str
    output: str
    options: RunOptions = field(default_factory=RunOptions)
    execution_started_at: datetime | None = None
    execution_completed_at: datetime | None = None
    execution_duration_seconds: float | None = None
    _file_loader: OutputFileLoader | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _files: tuple[OutputFile, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _files_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def files(self) -> tuple[OutputFile, ...]:
        """Return generated files, loading the protected listing once on demand."""

        with self._files_lock:
            cached = self._files
            if cached is None:
                loader = self._file_loader
                cached = () if loader is None else tuple(loader())
                object.__setattr__(self, "_files", cached)
                object.__setattr__(self, "_file_loader", None)
            return cached

    @property
    def artifacts(self) -> tuple[OutputFile, ...]:
        """Alias for :attr:`files`, matching agent-artifact terminology."""

        return self.files

    @property
    def mode(self) -> ConversationMode:
        """Return the model/research mode used for the completed turn."""

        return self.options.mode

    def _repr_markdown_(self) -> str:
        """Render the final Markdown response naturally in notebooks."""

        return self.output


RunOutcome: TypeAlias = NeedsInput | RunResult


__all__ = [
    "BillingTopUpApproval",
    "BillingTopUpRequest",
    "ConversationMode",
    "Dataset",
    "EventKind",
    "FileInput",
    "InputKind",
    "NeedsInput",
    "OutputFile",
    "ProgressPhase",
    "ProgressUpdate",
    "RunOptions",
    "RunOutcome",
    "RunResult",
    "RunStatus",
    "StreamOperation",
    "StreamUpdate",
    "ThinkEvent",
    "ThinkMessage",
    "UploadedFile",
    "UploadPhase",
    "UploadProgress",
]
