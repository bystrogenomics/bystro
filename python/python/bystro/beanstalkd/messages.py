from enum import Enum
from typing import get_type_hints

from msgspec import Struct, field

SubmissionID = str | int
BeanstalkJobID = int


class Event(str, Enum):
    """Beanstalkd Event"""

    PROGRESS = "progress"
    FAILED = "failed"
    STARTED = "started"
    COMPLETED = "completed"


class BaseMessage(Struct, frozen=True, rename="camel"):
    submission_id: SubmissionID

    @classmethod
    def keys_with_types(cls) -> dict:
        return get_type_hints(cls)


class SubmittedJobMessage(BaseMessage, frozen=True):
    event: Event = Event.STARTED


class CompletedJobMessage(BaseMessage, frozen=True):
    event: Event = Event.COMPLETED


class FailedJobMessage(BaseMessage, frozen=True):
    reason: str
    event: Event = Event.FAILED


class InvalidJobMessage(Struct, frozen=True, rename="camel"):
    # Invalid jobs that are invalid because the submission breaks serialization invariants
    # will not have a submission_id as that ID is held in the serialized data
    queue_id: BeanstalkJobID
    reason: str
    event: Event = Event.FAILED

    @classmethod
    def keys_with_types(cls) -> dict:
        return get_type_hints(cls)


class ProgressData(Struct):
    progress: int = 0
    skipped: int = 0


class ProgressMessage(BaseMessage, frozen=True):
    """Beanstalkd Message"""

    event: Event = Event.PROGRESS
    data: ProgressData = field(default_factory=ProgressData)
