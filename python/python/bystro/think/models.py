"""Public value objects for the Bystro Think SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

ConversationMode: TypeAlias = Literal["base", "plus", "plus2", "phd"]


class RunStatus(str, Enum):
    """Lifecycle state of a Think run."""

    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
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
    MESSAGE = "message"
    NEEDS_INPUT = "needs_input"
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


@dataclass(frozen=True, slots=True)
class ThinkEvent:
    """An ordered, SDK-level progress event."""

    sequence: int
    kind: EventKind
    run_id: str
    created_at: datetime
    message: str | None = None
    data: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NeedsInput:
    """A durable pause that must be resolved before work can continue."""

    run_id: str
    kind: InputKind
    prompt: str | None
    checkpoint_id: str | None
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Successful Think run result."""

    run_id: str
    output: str


RunOutcome: TypeAlias = NeedsInput | RunResult


__all__ = [
    "ConversationMode",
    "Dataset",
    "EventKind",
    "FileInput",
    "InputKind",
    "NeedsInput",
    "OutputFile",
    "RunOptions",
    "RunOutcome",
    "RunResult",
    "RunStatus",
    "ThinkEvent",
    "ThinkMessage",
    "UploadedFile",
    "UploadPhase",
    "UploadProgress",
]
