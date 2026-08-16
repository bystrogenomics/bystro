"""Bystro Think: submit agentic workloads from Python.

Typical usage::

    from bystro.think import ThinkClient, show_progress

    with ThinkClient.from_cached_login(on_event=show_progress) as client:
        run = client.submit_with_progress(
            "Summarize this cohort",
            files=["cohort.vcf"],
        )
        outcome = run.interact()
        print(outcome)
"""

from bystro.think.client import (
    DEFAULT_THINK_URL,
    Run,
    ThinkClient,
    UploadProgressCallback,
)
from bystro.think.context import (
    ArtifactReference,
    ContextArtifact,
    ContextTransform,
    MessageInput,
    MessageWithContext,
    PreviousConversation,
    add_artifact_context,
    add_genetic_context,
    add_previous_conversation_context,
    artifact_context,
    compose_context,
    genetic_context,
    previous_conversation_context,
)
from bystro.think.errors import (
    InputResponseError,
    RunCancelledError,
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
from bystro.think.progress import ProgressCallback, ProgressRenderer, show_progress

__all__ = [
    "ArtifactReference",
    "ConversationMode",
    "ContextArtifact",
    "ContextTransform",
    "DEFAULT_THINK_URL",
    "Dataset",
    "EventKind",
    "FileInput",
    "InputKind",
    "InputResponseError",
    "MessageInput",
    "MessageWithContext",
    "NeedsInput",
    "OutputFile",
    "ProgressPhase",
    "ProgressUpdate",
    "ProgressCallback",
    "PreviousConversation",
    "ProgressRenderer",
    "Run",
    "RunCancelledError",
    "RunOptions",
    "RunOutcome",
    "RunProtocolError",
    "RunRejectedError",
    "RunResult",
    "RunStatus",
    "StreamOperation",
    "StreamUpdate",
    "RunTimeoutError",
    "show_progress",
    "ThinkAuthenticationError",
    "ThinkBillingRequiredError",
    "ThinkClient",
    "ThinkConnectionError",
    "ThinkError",
    "ThinkEvent",
    "ThinkHTTPError",
    "ThinkMessage",
    "UploadedFile",
    "UploadPhase",
    "UploadProgress",
    "UploadProgressCallback",
    "add_artifact_context",
    "add_genetic_context",
    "add_previous_conversation_context",
    "artifact_context",
    "compose_context",
    "genetic_context",
    "previous_conversation_context",
]
