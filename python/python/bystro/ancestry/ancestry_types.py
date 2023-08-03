"""Classes for common shapes of data in ancestry."""
import re
from collections import Counter
from collections.abc import Sequence
from typing import TypeVar

import attrs
from attrs import field
from attrs.validators import ge, instance_of, le

# Attr classes can throw either TypeError or ValueError upon receipt
# of bad data.  Generally we won't care which, so define the union of
# the two exceptions to catch all validation errors.  Because we'll be
# using it in try blocks, this wants to be a tuple rather than a union.

AttrValidationError = (ValueError, TypeError)


#  It seems a bit overkill to have an attr class just for vcfs, but
#  the validator needs to be shared between classes. So it just lives
#  here as a pure python function with a weird type signature in order
#  to support what attr expects from its validators.

_VCF_REGEX = re.compile(
    r"""
    .*                        # anything
    \.vcf                     # .vcf extension
    (\.(gz|lz4|zstd|bzip2))?  # optional compression extension
    $                         # and nothing else afterwards
    """,
    re.VERBOSE,
)


def _vcf_validator(_self: object, _attribute: attrs.Attribute, vcf_path: str) -> None:
    if not isinstance(vcf_path, str):
        err_msg = f"vcf_path must be of type str, got: {vcf_path} instead"
        raise TypeError(err_msg)
    if not _VCF_REGEX.fullmatch(vcf_path):
        err_msg = (
            "vcf_path must be of extension .vcf, .vcf.gz, .vcf.lz4 .vcf.zstd or .vcf.bzip2, "
            f"got {vcf_path} instead"
        )
        raise ValueError(err_msg)


@attrs.frozen()
class AncestrySubmission:
    """Represent an incoming submission to the ancestry worker."""

    vcf_path: str = field(validator=_vcf_validator)


unit_float_validator = [
    instance_of(float),
    ge(0.0),
    le(1.0),
]


@attrs.frozen()
class ProbabilityInterval:
    """Represent an interval of probabilities."""

    # we need these to be literal floats for msgspec serialization, not numpy floats or anything else.
    lower_bound: float = field(converter=float, validator=unit_float_validator)
    upper_bound: float = field(converter=float, validator=unit_float_validator)

    def __attrs_post_init__(self) -> None:
        """Ensure interval is well-formed."""
        if self.lower_bound > self.upper_bound:
            err_msg = (
                f"Lower bound must be less than or equal to upper bound.  "
                f"Got: lower_bound={self.lower_bound}, upper_bound={self.upper_bound} instead."
            )
            raise ValueError(err_msg)


# NB: We might consider that a vector of ProbabilityIntervals should
# have additional validation properties, like that the sums of the
# lower bounds, upper bounds, or midpoints should be close to one.
# But constraints on the bounds don't hold in general (consider the
# vector of intervals [(0.4, 0.6), (0.4, 0.6)]), and we can't know how
# well the midpoints of the intervals reflect the point estimate in
# general, so we'll punt on this and assume it's the ML model's
# responsibility to give us scientifically sensible results.


# this definition is mildly ugly but the alternative is to
# generate it dynamically, which would be even worse...

ProbIntValidator = attrs.validators.instance_of(ProbabilityInterval)


@attrs.frozen(kw_only=True)
class PopulationVector:
    """A vector of probability intervals over populations.

    Represents model estimates of an individual's similarity to
    reference HapMap populations, with upper and lower bounds for each
    population.
    """

    ACB: ProbabilityInterval = field(validator=ProbIntValidator)
    ASW: ProbabilityInterval = field(validator=ProbIntValidator)
    BEB: ProbabilityInterval = field(validator=ProbIntValidator)
    CDX: ProbabilityInterval = field(validator=ProbIntValidator)
    CEU: ProbabilityInterval = field(validator=ProbIntValidator)
    CHB: ProbabilityInterval = field(validator=ProbIntValidator)
    CHS: ProbabilityInterval = field(validator=ProbIntValidator)
    CLM: ProbabilityInterval = field(validator=ProbIntValidator)
    ESN: ProbabilityInterval = field(validator=ProbIntValidator)
    FIN: ProbabilityInterval = field(validator=ProbIntValidator)
    GBR: ProbabilityInterval = field(validator=ProbIntValidator)
    GIH: ProbabilityInterval = field(validator=ProbIntValidator)
    GWD: ProbabilityInterval = field(validator=ProbIntValidator)
    IBS: ProbabilityInterval = field(validator=ProbIntValidator)
    ITU: ProbabilityInterval = field(validator=ProbIntValidator)
    JPT: ProbabilityInterval = field(validator=ProbIntValidator)
    KHV: ProbabilityInterval = field(validator=ProbIntValidator)
    LWK: ProbabilityInterval = field(validator=ProbIntValidator)
    MSL: ProbabilityInterval = field(validator=ProbIntValidator)
    MXL: ProbabilityInterval = field(validator=ProbIntValidator)
    PEL: ProbabilityInterval = field(validator=ProbIntValidator)
    PJL: ProbabilityInterval = field(validator=ProbIntValidator)
    PUR: ProbabilityInterval = field(validator=ProbIntValidator)
    STU: ProbabilityInterval = field(validator=ProbIntValidator)
    TSI: ProbabilityInterval = field(validator=ProbIntValidator)
    YRI: ProbabilityInterval = field(validator=ProbIntValidator)


@attrs.frozen(kw_only=True)
class SuperpopVector:
    """A vector of probability intervals for superpopulations.

    Represents model estimates of an individual's similarity to
    reference HapMap superpopulations, with upper and lower bounds for
    each population.

    """

    AFR: ProbabilityInterval = field(validator=ProbIntValidator)
    AMR: ProbabilityInterval = field(validator=ProbIntValidator)
    EAS: ProbabilityInterval = field(validator=ProbIntValidator)
    EUR: ProbabilityInterval = field(validator=ProbIntValidator)
    SAS: ProbabilityInterval = field(validator=ProbIntValidator)


@attrs.frozen(kw_only=True)
class AncestryResult:
    """An ancestry result from a sample.

    Represents ancestry model output for an individual study
    participant (identified by sample_id) with estimates for
    populations and superpopulations, and the overall fraction of
    expected variants found missing in the sample.
    """

    sample_id: str = field(validator=instance_of(str))
    populations: PopulationVector = field(validator=instance_of(PopulationVector))
    superpops: SuperpopVector = field(validator=instance_of(SuperpopVector))
    # needs to be literal float for msgspec
    missingness: float = field(converter=float, validator=unit_float_validator)


@attrs.frozen(kw_only=True)
class AncestryResponse:
    """An outgoing response from the ancestry worker.

    Represents ancestry model output for an entire study as a list of
    individual AncestryResults.

    """

    vcf_path: str = field(validator=_vcf_validator)
    results: list[AncestryResult] = field(validator=instance_of(list))

    @results.validator
    def _is_list_of_unique_ancestry_results(
        self, _attribute: attrs.Attribute, results: list[AncestryResult]
    ) -> None:
        for i, value in enumerate(results):
            if not isinstance(value, AncestryResult):
                err_msg = (
                    f"Expecting list of AncestryResults,"
                    f"at position {i} got {value} of type {type(value)} instead"
                )
                raise TypeError(err_msg)
        sample_ids = [result.sample_id for result in results]
        unique_sample_ids = set(sample_ids)
        if len(unique_sample_ids) < len(results):
            duplicates = _get_duplicates(sample_ids)
            err_msg = f"Expected unique sample_ids but found duplicated samples {duplicates}"
            raise ValueError(err_msg)


T = TypeVar("T")


def _get_duplicates(xs: Sequence[T]) -> set[T]:
    return {x for x, count in Counter(xs).items() if count > 1}
