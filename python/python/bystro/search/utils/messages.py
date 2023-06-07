from bystro.beanstalkd.messages import BaseMessage, Struct
from bystro.search.utils.annotation import AnnotationOutputs


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


class SaveJobResults(Struct, frozen=True):
    outputFileNames: AnnotationOutputs


class SaveJobCompleteMessage(BaseMessage, frozen=True):
    """Beanstalkd Job data"""

    results: SaveJobResults
