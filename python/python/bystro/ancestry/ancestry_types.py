"""Classes for common shapes of data in ancestry."""
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


def _vcf_validator(_self: object, _attribute: attrs.Attribute, value: str) -> None:
    if not isinstance(value, str):
        err_msg = f"vcf_path must be of type str, got: {value} instead"
        raise TypeError(err_msg)
    if "vcf" not in value:
        err_msg = f"Expected vcf_path ending in '.vcf', got {value} instead"
        raise ValueError(err_msg)


@attrs.frozen()
class AncestrySubmission:
    """Represent an incoming submission to the ancestry worker."""

    vcf_path: str = attrs.field(validator=_vcf_validator)


unit_float_validator = [
    instance_of(float),
    ge(0.0),
    le(1.0),
]


@attrs.frozen()
class ProbabilityInterval:
    """Represent an interval of probabilities."""

    lower_bound: float = field(validator=unit_float_validator)
    upper_bound: float = field(
        validator=[
            instance_of(float),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ]
    )

    def __attrs_post_init__(self) -> None:
        """Ensure interval is well-formed."""
        if self.lower_bound > self.upper_bound:
            err_msg = f"""Lower bound must be less than or equal to upper bound.
            Got: lower_bound={self.lower_bound}, upper_bound={self.upper_bound} instead."""
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
    """A vector of probability intervals for populations."""

    ACB: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    ASW: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    BEB: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    CDX: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    CEU: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    CHB: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    CHS: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    CLM: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    ESN: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    FIN: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    GBR: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    GIH: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    GWD: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    IBS: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    ITU: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    JPT: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    KHV: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    LWK: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    MAG: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    MSL: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    MXL: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    PEL: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    PJL: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    PUR: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    STU: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    TSI: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    YRI: ProbabilityInterval = attrs.field(validator=ProbIntValidator)


@attrs.frozen(kw_only=True)
class SuperpopVector:
    """A vector of probability intervals for superpopulations."""

    AFR: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    AMR: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    EAS: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    EUR: ProbabilityInterval = attrs.field(validator=ProbIntValidator)
    SAS: ProbabilityInterval = attrs.field(validator=ProbIntValidator)


@attrs.frozen(kw_only=True)
class AncestryResult:
    """An ancestry result from a sample."""

    sample_id: str = field(validator=instance_of(str))
    populations: PopulationVector = field(validator=instance_of(PopulationVector))
    superpops: SuperpopVector = field(validator=instance_of(SuperpopVector))
    missingness: float = field(validator=unit_float_validator)


@attrs.frozen(kw_only=True)
class AncestryResponse:
    """An outgoing response from the ancestry worker."""

    vcf_path: str = attrs.field(validator=_vcf_validator)
    results: list[AncestryResult] = field(validator=instance_of(list))

    @results.validator
    def _is_list_of_unique_ancestry_results(
        self, _attribute: attrs.Attribute, results: list[AncestryResult]
    ) -> None:
        for i, value in enumerate(results):
            if not isinstance(value, AncestryResult):
                err_msg = f"Expecting list of AncestryResults, at position {i} got {value} instead"
                raise TypeError(err_msg)
        sample_ids = [result.sample_id for result in results]
        unique_sample_ids = set(sample_ids)
        if len(unique_sample_ids) < len(results):
            duplicates = _get_duplicates(sample_ids)
            err_msg = f"Expected unique sample ids but found duplicated samples {duplicates}"
            raise ValueError(err_msg)


T = TypeVar("T")


def _get_duplicates(xs: Sequence[T]) -> set[T]:
    return {x for x, count in Counter(xs).items() if count > 1}
