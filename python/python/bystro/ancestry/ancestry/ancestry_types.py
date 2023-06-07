"""Pydantic validators for common shapes of data in ancestry."""


from typing import Any

from pydantic import BaseModel, Extra, Field, root_validator, validator


class AncestrySubmission(BaseModel, extra=Extra.forbid):
    """Represent an incoming submission to the ancestry worker."""

    vcf_path: str


class ProbabilityInterval(BaseModel, extra=Extra.forbid):
    """Represent an interval of probabilities."""

    lower_bound: float = Field(ge=0, le=1)
    upper_bound: float = Field(ge=0, le=1)

    @validator("upper_bound")
    def _interval_is_valid(
        cls: "ProbabilityInterval",  # noqa: N805 (false positive on cls name)
        upper_bound: float,
        values: dict[str, Any],
    ) -> float:
        """Ensure interval is non-empty."""
        lower_bound = values["lower_bound"]
        if not lower_bound <= upper_bound:
            err_msg = (
                f"Must have lower_bound <= upper_bound:"
                f" got (lower_bound={lower_bound}, upper_bound={upper_bound}) instead."
            )
            raise ValueError(err_msg)
        return upper_bound


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
class PopulationVector(BaseModel, extra=Extra.forbid):
    """A vector of probability intervals for populations."""

    ACB: ProbabilityInterval
    ASW: ProbabilityInterval
    BEB: ProbabilityInterval
    CDX: ProbabilityInterval
    CEU: ProbabilityInterval
    CHB: ProbabilityInterval
    CHS: ProbabilityInterval
    CLM: ProbabilityInterval
    ESN: ProbabilityInterval
    FIN: ProbabilityInterval
    GBR: ProbabilityInterval
    GIH: ProbabilityInterval
    GWD: ProbabilityInterval
    IBS: ProbabilityInterval
    ITU: ProbabilityInterval
    JPT: ProbabilityInterval
    KHV: ProbabilityInterval
    LWK: ProbabilityInterval
    MAG: ProbabilityInterval
    MSL: ProbabilityInterval
    MXL: ProbabilityInterval
    PEL: ProbabilityInterval
    PJL: ProbabilityInterval
    PUR: ProbabilityInterval
    STU: ProbabilityInterval
    TSI: ProbabilityInterval
    YRI: ProbabilityInterval


class SuperpopVector(BaseModel, extra=Extra.forbid):
    """A vector of probability intervals for superpopulations."""

    AFR: ProbabilityInterval
    AMR: ProbabilityInterval
    EAS: ProbabilityInterval
    EUR: ProbabilityInterval
    SAS: ProbabilityInterval


class AncestryResult(BaseModel, extra=Extra.forbid):
    """An ancestry result from a sample."""

    sample_id: str
    populations: PopulationVector
    superpops: SuperpopVector
    missingness: float = Field(ge=0, le=1)


class AncestryResponse(BaseModel, extra=Extra.forbid):
    """An outgoing response from the ancestry worker."""

    vcf_path: str
    results: list[AncestryResult]

    @root_validator
    def _ensure_sample_ids_unique(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        results = values["results"]
        sample_ids = [result.sample_id for result in results]
        unique_sample_ids = set(sample_ids)
        if len(unique_sample_ids) < len(sample_ids):
            duplicates = [sid for sid in unique_sample_ids if sample_ids.count(sid) > 1]
            err_msg = f"Expected unique sample ids but found duplicated samples {duplicates}"
            raise ValueError(err_msg)
        return values
