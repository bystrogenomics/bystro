from bystro.beanstalkd.messages import (
    BaseMessage,
    SubmittedJobMessage,
    CompletedJobMessage,
    Struct,
)
from bystro.search.utils.annotation import AnnotationOutputs


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


class BinomialMafFilter(
    Struct, frozen=True, tag="binomMaf", tag_field="key", forbid_unknown_fields=True
):
    """
    Example: {'critValue': '.05', 'estimates': ['gnomad.genomes.af', 'gnomad.exomes.af'],
             'privateMaf': 0.0001, 'snpOnly': True, 'numSamples': 3, 'key': 'binomialMaf'}
    """

    privateMaf: float
    snpOnly: bool
    numSamples: int
    estimates: list[str]
    critValue: float | None = 0.05


class HWEFilter(
    Struct, frozen=True, tag="hwe", tag_field="key", forbid_unknown_fields=True
):
    """
    Example: {'critValue': '.05', 'numSamples': 3, 'key': 'hwe'}
    """

    numSamples: int
    critValue: float | None = 0.05


class SaveJobData(BaseMessage, frozen=True):
    """Data for SaveFromQuery jobs received from beanstalkd"""

    assembly: str
    queryBody: dict
    indexName: str
    outputBasePath: str
    fieldNames: list[str]
    pipeline: list[BinomialMafFilter | HWEFilter] | None = None


class SaveJobSubmitMessage(SubmittedJobMessage, frozen=True, kw_only=True):
    jobConfig: dict


class SaveJobResults(Struct, frozen=True):
    outputFileNames: AnnotationOutputs


class SaveJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    results: SaveJobResults
