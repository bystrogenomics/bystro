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
    Filter out rows that are extreme outliers, comparing the in-sample estimate of allele frequency,
    the `sampleMaf` from the indexed annotation, to the "true" population allele freuqency estimate
    from the `estimates` parameter.

    This filters operates on the assumption that the alleles are binomially distributed,
    and uses the normal approximation to the binomial distribution to identify outliers.

    A variant is considered an outlier if the `sampleMaf` is outside the `1-(critValue*2)`
    rejection region. A 2 tailed test is used.

    Since very rare mutations we may have unreliable population estimates, if the in-sample
    allele frequency `sampelMaf` is less than the `privateMaf` threshold,
    we retain the variant regardless of whether it is an outlier.

    Parameters
    ----------
    privateMaf : float
        The rare frequency threshold. If the sampleMaf is less than this value, the variant is retained
    snpOnly : bool
        Whether or not only single nucleotide variants (SNPs) should be considered
    numSamples : int
        The number of samples in the population
    estimates : list[str]
        The names of the columns that contain the population allele frequency estimates
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
    A Hardy-Weinberg Equilibrium (HWE) filter,
    which filters out variants that are not in HWE.

    Parameters
    ----------
    numSamples : int
        Number of samples in the population
    critValue : float, optional
        The critical value for the chi-squared test.
        Default: 0.05
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
