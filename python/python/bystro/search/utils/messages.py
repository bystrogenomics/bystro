from typing import get_type_hints

from enum import Enum

from msgspec import Struct, field

from bystro.search.utils.annotation import AnnotationOutputs

SUBMISSION_ID = str | int
BEANSTALK_JOB_ID = int


class BaseMessage(Struct, frozen=True):
    submissionID: SUBMISSION_ID

    @classmethod
    def keys_with_types(cls) -> dict:
        return get_type_hints(cls)


class FailedJobMessage(Struct, frozen=True):
    submissionID: SUBMISSION_ID
    reason: str


class InvalideJobMessage(Struct, frozen=True):
    # Invalid jobs that are invalid because the submission breaks serialization invariants will not have a submissionID
    # as that ID is held in the serialized data
    queueID: BEANSTALK_JOB_ID
    reason: str


class Event(str, Enum):
    """Beanstalkd Event"""

    PROGRESS = "progress"
    FAILED = "failed"
    STARTED = "started"
    COMPLETED = "completed"


class ProgressData(Struct):
    progress: int = 0
    skipped: int = 0


class ProgressMessage(BaseMessage, frozen=True):
    """Beanstalkd Message"""

    event: str = Event.PROGRESS
    data: ProgressData = field(default_factory=ProgressData)


class IndexJobData(BaseMessage, frozen=True):
    inputDir: str
    inputFileNames: AnnotationOutputs
    indexName: str
    assembly: str
    fieldNames: list[str] | None = None
    indexConfig: dict | None = None


class IndexJobResults(Struct, frozen=True):
    indexConfig: dict
    fieldNames: list


class IndexJobCompleteMessage(BaseMessage, frozen=True):
    """Beanstalkd Job data"""

    results: IndexJobResults


class SaveJobData(BaseMessage, frozen=True):
    """Beanstalkd Job data"""

    assembly: str
    queryBody: dict
    indexName: str
    inputQuery: str
    outputBasePath: str
    fieldNames: list[str]
    pipeline: dict | None = None
    indexConfig: dict | None = None


class SaveJobSubmitMessage(BaseMessage, frozen=True):
    """Beanstalkd Job data"""

    jobConfig: dict


class SaveJobResults(Struct):
    outputFileNames: AnnotationOutputs


class SaveJobCompleteMessage(BaseMessage, frozen=True):
    """Beanstalkd Job data"""

    results: SaveJobResults
