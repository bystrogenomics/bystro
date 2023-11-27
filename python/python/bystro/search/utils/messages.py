from typing import Callable

from bystro.beanstalkd.messages import (
    BaseMessage,
    SubmittedJobMessage,
    CompletedJobMessage,
    Struct,
)
from bystro.search.utils.annotation import AnnotationOutputs
from bystro.search.save.hwe import HWEFilter


class IndexJobData(BaseMessage, frozen=True):
    """Data for Indexing jobs received from beanstalkd"""

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
    Struct,
    frozen=True,
    tag="binomMaf",
    tag_field="key",
    forbid_unknown_fields=True,
    rename="camel",
):
    """
    Filter out rows that are extreme outliers, comparing the in-sample estimate of allele frequency,
    the `sampleMaf` from the indexed annotation, to the "true" population allele frequency estimate
    from the `estimates` parameter.

    This filters operates on the assumption that the alleles are binomially distributed,
    and uses the normal approximation to the binomial distribution to identify outliers.

    A variant is considered an outlier if the `sampleMaf` is outside the `1-(critValue*2)`
    rejection region. A 2 tailed test is used.

    Since very rare mutations we may have unreliable population estimates, if the in-sample
    allele frequency `sampleMaf` is less than the `privateMaf` threshold,
    we retain the variant regardless of whether it is an outlier.

    Parameters
    ----------
    private_maf : float
        The rare frequency threshold. If the sampleMaf is less than this value, the variant is retained
    snp_only : bool
        Whether or not only single nucleotide variants (SNPs) should be considered
    num_samples : int
        The number of samples in the population
    estimates : list[str]
        The names of the columns that contain the population allele frequency estimates
    crit_value : float, optional
        The critical value defining the rejection region. A 2 tail test is used,
        so the rejection region is `1-(critValue*2)`.
        Default: 0.025
    """

    private_maf: float
    snp_only: bool
    num_samples: int
    estimates: list[str]
    crit_value: float | None = 0.025

    def make_filter(self) -> Callable[[dict], bool] | None:
        pass


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
