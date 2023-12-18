from bystro.beanstalkd.messages import (
    BaseMessage,
    SubmittedJobMessage,
    CompletedJobMessage,
    Struct,
)
from bystro.search.utils.annotation import AnnotationOutputs
from bystro.search.save.hwe import HWEFilter
from bystro.search.save.binomial_maf import BinomialMafFilter


class IndexJobData(BaseMessage, frozen=True):
    """Data for SaveFromQuery jobs received from beanstalkd"""

    inputDir: str
    inputFileNames: AnnotationOutputs
    indexName: str
    assembly: str
    fieldNames: list[str] | None = None
    indexConfig: dict | None = None


class IndexJobResults(Struct, frozen=True):
    indexConfig: dict
    fieldNames: list


class IndexJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    results: IndexJobResults


PipelineType = list[BinomialMafFilter | HWEFilter] | None


class SaveJobData(BaseMessage, frozen=True):
    """Data for SaveFromQuery jobs received from beanstalkd"""

    assembly: str
    queryBody: dict
    indexName: str
    outputBasePath: str
    fieldNames: list[str]
    pipeline: PipelineType = None


class SaveJobSubmitMessage(SubmittedJobMessage, frozen=True, kw_only=True):
    jobConfig: dict


class SaveJobResults(Struct, frozen=True):
    outputFileNames: AnnotationOutputs


class SaveJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    results: SaveJobResults
