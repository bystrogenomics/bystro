"""Classes for common shapes of data in ancestry."""

import attr
from attr import field
from attr.validators import ge, instance_of, le

# Attr classes can throw either TypeError or ValueError upon receipt
# of bad data.  Generally we won't care which, so define the union of
# the two exceptions to catch all validation errors.  Because we'll be
# using it in try blocks, this wants to be a tuple rather than a union.

AttrValidationError = (ValueError, TypeError)


#  It seems a bit overkill to have an attr class just for vcfs, but
#  the validator needs to be shared between classes. So it just lives
#  here as a pure python function with a weird type signature in order
#  to support what attr expects from its validators.


def _vcf_validator(_self: object, _attribute: attr.Attribute, value: str) -> None:
    if not isinstance(value, str):
        err_msg = f"vcf_path must be of type str, got: {value} instead"
        raise TypeError(err_msg)
    if not value.endswith(".vcf"):
        err_msg = f"Expected vcf_path ending in '.vcf', got {value} instead"
        raise ValueError(err_msg)


@attr.s(frozen=True)
class AncestrySubmission:
    """Represent an incoming submission to the ancestry worker."""

    vcf_path: str = attr.field(validator=_vcf_validator)


unit_float_validator = [
    instance_of(float),
    ge(0.0),
    le(1.0),
]


@attr.s(frozen=True)
class ProbabilityInterval:
    """Represent an interval of probabilities."""

    lower_bound: float = attr.field(validator=unit_float_validator)
    upper_bound: float = attr.ib(
        validator=[
            instance_of(float),
            attr.validators.ge(0.0),
            attr.validators.le(1.0),
        ]
    )

    def __attrs_post_init__(self) -> None:
        """Ensure interval is well-formed."""
        if not self.lower_bound <= self.upper_bound:
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

ProbIntValidator = attr.validators.instance_of(ProbabilityInterval)


@attr.s(kw_only=True, frozen=True)
class PopulationVector:
    """A vector of probability intervals for populations."""

    ACB: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    ASW: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    BEB: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    CDX: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    CEU: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    CHB: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    CHS: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    CLM: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    ESN: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    FIN: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    GBR: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    GIH: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    GWD: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    IBS: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    ITU: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    JPT: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    KHV: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    LWK: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    MAG: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    MSL: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    MXL: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    PEL: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    PJL: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    PUR: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    STU: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    TSI: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    YRI: ProbabilityInterval = attr.field(validator=ProbIntValidator)


@attr.s(kw_only=True, frozen=True)
class SuperpopVector:
    """A vector of probability intervals for superpopulations."""

    AFR: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    AMR: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    EAS: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    EUR: ProbabilityInterval = attr.field(validator=ProbIntValidator)
    SAS: ProbabilityInterval = attr.field(validator=ProbIntValidator)


@attr.s(kw_only=True, frozen=True)
class AncestryResult:
    """An ancestry result from a sample."""

    sample_id: str = field(validator=instance_of(str))
    populations: PopulationVector = field(validator=instance_of(PopulationVector))
    superpops: SuperpopVector = field(validator=instance_of(SuperpopVector))
    missingness: float = field(validator=unit_float_validator)


@attr.s(kw_only=True, frozen=True)
class AncestryResponse:
    """An outgoing response from the ancestry worker."""

    vcf_path: str = attr.field(validator=_vcf_validator)
    results: list[AncestryResult] = field(validator=instance_of(list))

    @results.validator
    def _is_list_of_ancestry_results(
        self, _attribute: attr.Attribute, values: list[AncestryResult]
    ) -> None:
        for i, value in enumerate(values):
            if not isinstance(value, AncestryResult):
                err_msg = f"Expecting list of AncestryResults, at position {i} got {value} instead"
                raise TypeError(err_msg)

    def __attrs_post_init__(self) -> None:
        """Ensure sample_ids are unique."""
        sample_ids = [result.sample_id for result in self.results]
        unique_sample_ids = set(sample_ids)
        if len(unique_sample_ids) < len(sample_ids):
            duplicates = [sid for sid in unique_sample_ids if sample_ids.count(sid) > 1]
            err_msg = f"Expected unique sample ids but found duplicated samples {duplicates}"
            raise ValueError(err_msg)
