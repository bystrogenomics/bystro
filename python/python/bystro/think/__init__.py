"""Bystro Think: submit agentic workloads from Python.

Typical usage::

    from bystro.api import auth
    from bystro.think import NeedsInput, ThinkClient

    auth.login("you@example.com", "password")
    with ThinkClient() as client:
        run = client.submit("Summarize this cohort", files=["cohort.vcf"])
        outcome = run.wait()
        if isinstance(outcome, NeedsInput):
            outcome = run.respond("Use the case_control column").wait()
        print(outcome.output)
"""

from bystro.think.client import (
    DEFAULT_THINK_URL,
    ProgressCallback,
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
    "ProgressCallback",
    "PreviousConversation",
    "Run",
    "RunOptions",
    "RunOutcome",
    "RunProtocolError",
    "RunRejectedError",
    "RunResult",
    "RunStatus",
    "RunTimeoutError",
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
