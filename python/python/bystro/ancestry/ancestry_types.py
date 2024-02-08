"""Classes for common shapes of data in ancestry."""
from msgspec import Struct

LOWER_UNIT_BOUND = 0.0
UPPER_UNIT_BOUND = 1.0

class ProbabilityInterval(Struct, rename="camel"):
    """Represent an interval of probabilities."""

    lower_bound: float
    upper_bound: float

    # Currently msgspec constraint bounds don't seem to integrate with
    # mypy or pyright, so we'll have to do this manually.
    # See tracking issue https://github.com/jcrist/msgspec/issues/177
    def __post_init__(self):
        # Due to PEP 484, types checkers do not distinguish between float and int
        if not isinstance(self.lower_bound, float):
            raise TypeError(f"lower_bound must be a float, not {type(self.lower_bound)}")
        if not isinstance(self.upper_bound, float):
            raise TypeError(f"upper_bound must be a float, not {type(self.upper_bound)}")

        if self.lower_bound < LOWER_UNIT_BOUND:
            raise TypeError(f"lower_bound must be >= {LOWER_UNIT_BOUND}")

        if self.upper_bound > UPPER_UNIT_BOUND:
            raise TypeError(f"upper_bound must be <= {UPPER_UNIT_BOUND}")


# NB: We might consider that a vector of ProbabilityIntervals should
# have additional validation properties, like that the sums of the
# lower bounds, upper bounds, or midpoints should be close to one.
# But constraints on the bounds don't hold in general (consider the
# vector of intervals [(0.4, 0.6), (0.4, 0.6)]), and we can't know how
# well the midpoints of the intervals reflect the point estimate in
# general, so we'll punt on this and assume it's the ML model's
# responsibility to give us scientifically sensible results.


class PopulationVector(Struct, kw_only=True):
    """A vector of probability intervals over populations.

    Represents model estimates of an individual's similarity to
    reference HapMap populations, with upper and lower bounds for each
    population.
    """

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
    MSL: ProbabilityInterval
    MXL: ProbabilityInterval
    PEL: ProbabilityInterval
    PJL: ProbabilityInterval
    PUR: ProbabilityInterval
    STU: ProbabilityInterval
    TSI: ProbabilityInterval
    YRI: ProbabilityInterval


class SuperpopVector(Struct, kw_only=True):
    """A vector of probability intervals for superpopulations.

    Represents model estimates of an individual's similarity to
    reference HapMap superpopulations, with upper and lower bounds for
    each population.

    """

    AFR: ProbabilityInterval
    AMR: ProbabilityInterval
    EAS: ProbabilityInterval
    EUR: ProbabilityInterval
    SAS: ProbabilityInterval


class AncestryTopHit(Struct):
    """
    The top hit for a sample, with the max value (a probability) and the population(s) corresponding
    """

    probability: float
    populations: list[str]

    def __post_init__(self):
        if not isinstance(self.probability, float):
            raise TypeError(f"probability must be a float, not {type(self.probability)}")

        if self.probability < LOWER_UNIT_BOUND or self.probability > UPPER_UNIT_BOUND:
            raise TypeError(f"probability must be between {LOWER_UNIT_BOUND} and {UPPER_UNIT_BOUND}")


class AncestryScoresOneSample(Struct, frozen=True, rename="camel"):
    """An ancestry result for a sample.

    Represents ancestry model output for an individual study
    participant (identified by sample_id) with estimates for
    populations and superpopulations, and the overall fraction of
    expected variants found missing in the sample.
    """

    sample_id: str
    top_hit: AncestryTopHit
    populations: PopulationVector
    superpops: SuperpopVector
    missingness: float

    def __post_init__(self):
        if not isinstance(self.missingness, float):
            raise TypeError(f"missingness must be a float, not {type(self.missingness)}")

        if self.missingness < LOWER_UNIT_BOUND or self.missingness > UPPER_UNIT_BOUND:
            raise TypeError(f"missingness must be between {LOWER_UNIT_BOUND} and {UPPER_UNIT_BOUND}")


class AncestryResults(Struct, frozen=True):
    """An outgoing response from the ancestry worker.

    Represents ancestry model output for an entire study as a list of
    individual AncestryResults.

    """

    results: list[AncestryScoresOneSample]
    pcs: dict[str, list[float]]
