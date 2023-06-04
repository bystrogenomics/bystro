from typing import get_type_hints

from enum import Enum

from msgspec import Struct

from bystro.search.utils.annotation import AnnotationOutputs

class BaseMessage(Struct, frozen=True):
    submissionID: str

    @classmethod
    def keys_with_types(cls) -> dict:
        return get_type_hints(cls)

class FailedMessage(BaseMessage, frozen=True):
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
    data: ProgressData | str | None = None

class IndexJobData(BaseMessage, frozen=True):
    inputDir: str
    inputFileNames: AnnotationOutputs
    indexName: str
    assembly: str
    fieldNames: list[str] | None = None
    indexConfig: dict | None = None

class IndexResultData(BaseMessage, frozen=True):
    indexConfig: dict
    fieldNames: list

class SaveJobData(BaseMessage, frozen=True):
    """Beanstalkd Job data"""
    submissionID: str
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
    outputFileNames: list[str]

class SaveJobCompleteMessage(BaseMessage, frozen=True):
    """Beanstalkd Job data"""
    results: SaveJobResults