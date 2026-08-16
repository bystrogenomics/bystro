"""Composable context builders for Think messages.

The object retains structured references for transport. ``to_xml()`` exposes
a safely escaped semantic preview. Think resolves artifacts for the
authenticated user, and retrieval tools enforce dataset and conversation
ownership before returning referenced data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
import html
from typing import TypeAlias, TypeVar

from bystro.think.models import Dataset, UploadedFile


@dataclass(frozen=True, slots=True)
class PreviousConversation:
    """Reference to a durable Think conversation."""

    id: str  # noqa: A003 - mirrors the conversation API field
    name: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("conversation id cannot be empty")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Reference to an existing personal input artifact."""

    id: str  # noqa: A003 - mirrors the artifact API field

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("artifact id cannot be empty")


ContextArtifact: TypeAlias = UploadedFile | ArtifactReference


@dataclass(frozen=True, slots=True)
class MessageWithContext:
    """An immutable prompt plus Bystro context references."""

    prompt: str
    datasets: tuple[Dataset, ...] = ()
    conversations: tuple[PreviousConversation, ...] = ()
    artifacts: tuple[ContextArtifact, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")

    def to_xml(self) -> str:
        """Render a safe preview of the semantic prompt context."""

        sections: list[str] = []
        if self.datasets:
            entries: list[str] = []
            for dataset in self.datasets:
                attrs = [
                    f'name="{_attribute(dataset.name)}"',
                    f'id="{_attribute(dataset.id)}"',
                ]
                if dataset.assembly:
                    attrs.append(f'assembly="{_attribute(dataset.assembly)}"')
                entries.append(f"<dataset {' '.join(attrs)}/>")
            sections.append("<bystro_datasets>\n" + "\n".join(entries) + "\n</bystro_datasets>")
        if self.artifacts:
            entries = []
            for artifact in self.artifacts:
                if isinstance(artifact, UploadedFile):
                    attrs = [
                        f'id="{_attribute(artifact.id)}"',
                        f'name="{_attribute(artifact.name)}"',
                        f'displayName="{_attribute(artifact.display_name)}"',
                        f'size="{_attribute(artifact.size)}"',
                    ]
                    if artifact.mime:
                        attrs.append(f'mime="{_attribute(artifact.mime)}"')
                    entries.append(f"<file {' '.join(attrs)}/>")
                else:
                    entries.append(f'<file id="{_attribute(artifact.id)}"/>')
            sections.append("<input_artifacts>\n" + "\n".join(entries) + "\n</input_artifacts>")
        if self.conversations:
            entries = []
            for conversation in self.conversations:
                attrs = [f'id="{_attribute(conversation.id)}"']
                if conversation.name:
                    attrs.append(f'name="{_attribute(conversation.name)}"')
                entries.append(f"<conversation {' '.join(attrs)}/>")
            sections.append(
                "<context_conversations>\n"
                + "\n".join(entries)
                + "\n</context_conversations>"
            )
        if not sections:
            return self.prompt
        return f"{self.prompt}\n\nThis request references:\n\n" + "\n\n".join(sections)

    def __str__(self) -> str:
        return self.to_xml()


MessageInput: TypeAlias = str | MessageWithContext
ContextTransform: TypeAlias = Callable[[MessageInput], MessageWithContext]
_ContextItem = TypeVar("_ContextItem")


def _attribute(value: object) -> str:
    return html.escape(str(value), quote=True)


def _message(value: MessageInput) -> MessageWithContext:
    return value if isinstance(value, MessageWithContext) else MessageWithContext(value)


def _upsert_by_id(
    items: tuple[_ContextItem, ...],
    item: _ContextItem,
    identifier: Callable[[_ContextItem], str],
) -> tuple[_ContextItem, ...]:
    item_id = identifier(item)
    return tuple(existing for existing in items if identifier(existing) != item_id) + (item,)


def add_genetic_context(
    job_id: str,
    message: MessageInput,
    *,
    name: str | None = None,
    assembly: str | None = None,
) -> MessageWithContext:
    """Attach a Bystro annotation job to a prompt or context message."""

    current = _message(message)
    dataset = Dataset(id=job_id, name=name or job_id, assembly=assembly)
    return replace(
        current,
        datasets=_upsert_by_id(current.datasets, dataset, lambda item: item.id),
    )


def add_previous_conversation_context(
    thread_id: str,
    message: MessageInput,
    *,
    name: str | None = None,
) -> MessageWithContext:
    """Attach a prior Think conversation for on-demand agent retrieval."""

    current = _message(message)
    conversation = PreviousConversation(id=thread_id, name=name)
    return replace(
        current,
        conversations=_upsert_by_id(
            current.conversations,
            conversation,
            lambda item: item.id,
        ),
    )


def add_artifact_context(
    artifact: UploadedFile | ArtifactReference | str,
    message: MessageInput,
) -> MessageWithContext:
    """Attach an uploaded artifact or a durable artifact id."""

    current = _message(message)
    reference: ContextArtifact = (
        ArtifactReference(artifact) if isinstance(artifact, str) else artifact
    )
    return replace(
        current,
        artifacts=_upsert_by_id(current.artifacts, reference, lambda item: item.id),
    )


def genetic_context(
    job_id: str,
    *,
    name: str | None = None,
    assembly: str | None = None,
) -> ContextTransform:
    """Return a reusable transform that adds a genetic job context."""

    return lambda message: add_genetic_context(
        job_id,
        message,
        name=name,
        assembly=assembly,
    )


def previous_conversation_context(
    thread_id: str,
    *,
    name: str | None = None,
) -> ContextTransform:
    """Return a reusable transform that adds a prior conversation."""

    return lambda message: add_previous_conversation_context(
        thread_id,
        message,
        name=name,
    )


def artifact_context(
    artifact: UploadedFile | ArtifactReference | str,
) -> ContextTransform:
    """Return a reusable transform that adds an input artifact."""

    return lambda message: add_artifact_context(artifact, message)


def compose_context(*transforms: ContextTransform) -> ContextTransform:
    """Compose context transforms from left to right."""

    def apply(message: MessageInput) -> MessageWithContext:
        current = _message(message)
        for transform in transforms:
            current = transform(current)
        return current

    return apply


__all__ = [
    "ArtifactReference",
    "ContextArtifact",
    "ContextTransform",
    "MessageInput",
    "MessageWithContext",
    "PreviousConversation",
    "add_artifact_context",
    "add_genetic_context",
    "add_previous_conversation_context",
    "artifact_context",
    "compose_context",
    "genetic_context",
    "previous_conversation_context",
]
