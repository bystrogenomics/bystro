"""Test ancestry_types.py."""

import re

import msgspec
import pytest

from bystro.ancestry.ancestry_types import (
    AncestryResults,
    AncestryScoresOneSample,
    AncestryTopHit,
    PopulationVector,
    ProbabilityInterval,
    SuperpopVector,
)
from bystro.ancestry.train import POPS, SUPERPOPS


# ruff: noqa: E721

# In several tests we explicitly check that a value `is float` rather
# than use the more pythonic `isinstance(value, float)`.  We make
# these explicit checks in order to ensure that such values are raw
# floats and not np.float64's, which can't easily be deserialized.
# But the former method of checking raises E71 errors, which are
# exempted from the linter on a file-wide basis above.

prob_int = ProbabilityInterval(lower_bound=0.0, upper_bound=1.0)


pop_kwargs = {pop: prob_int for pop in POPS}
superpop_kwargs = {pop: prob_int for pop in SUPERPOPS}


def test_expected_population_vector():
    """Ensure that the expected populations are found."""
    expected = {
        "ACB": prob_int,
        "ASW": prob_int,
        "BEB": prob_int,
        "CDX": prob_int,
        "CEU": prob_int,
        "CHB": prob_int,
        "CHS": prob_int,
        "CLM": prob_int,
        "ESN": prob_int,
        "FIN": prob_int,
        "GBR": prob_int,
        "GIH": prob_int,
        "GWD": prob_int,
        "IBS": prob_int,
        "ITU": prob_int,
        "JPT": prob_int,
        "KHV": prob_int,
        "LWK": prob_int,
        "MSL": prob_int,
        "MXL": prob_int,
        "PEL": prob_int,
        "PJL": prob_int,
        "PUR": prob_int,
        "STU": prob_int,
        "TSI": prob_int,
        "YRI": prob_int,
    }
    assert msgspec.structs.asdict(PopulationVector(**pop_kwargs)) == expected


def test_ProbabilityInterval_accepts_valid_bounds() -> None:
    """Ensure we can instantiate, validate ProbabilityInterval correctly."""
    prob_int = ProbabilityInterval(lower_bound=0.1, upper_bound=0.9)
    assert type(prob_int.lower_bound) is float
    assert type(prob_int.upper_bound) is float


def test_ProbabilityInterval_rejects_invalid_lower_bound() -> None:
    with pytest.raises(TypeError, match="lower_bound must be >= 0.0"):
        ProbabilityInterval(lower_bound=-0.1, upper_bound=0.9)


def test_ProbabilityInterval_rejects_invalid_upper_bound() -> None:
    with pytest.raises(TypeError, match="upper_bound must be <= 1.0"):
        ProbabilityInterval(lower_bound=0.1, upper_bound=1.1)


def test_ProbabilityInterval_rejects_ints() -> None:
    with pytest.raises(TypeError, match="lower_bound must be a float, not <class 'int'>"):
        ProbabilityInterval(lower_bound=int(0), upper_bound=1.0)

    with pytest.raises(TypeError, match="upper_bound must be a float, not <class 'int'>"):
        ProbabilityInterval(lower_bound=0.0, upper_bound=int(1))


def test_PopulationVector_accepts_valid_args() -> None:
    """Ensure we can instantiate, validate PopulationVector correctly."""
    PopulationVector(**pop_kwargs)


def test_PopulationVector_rejects_missing_key() -> None:
    pop_kwargs_with_missing_key = pop_kwargs.copy()
    del pop_kwargs_with_missing_key["ACB"]
    with pytest.raises(TypeError, match="Missing required argument 'ACB'"):
        PopulationVector(**pop_kwargs_with_missing_key)


def test_PopulationVector_rejects_extra_key() -> None:
    pop_kwargs_with_extra_key = pop_kwargs.copy()
    pop_kwargs_with_extra_key["FOO"] = prob_int
    with pytest.raises(TypeError, match="Unexpected keyword argument 'FOO'"):
        PopulationVector(**pop_kwargs_with_extra_key)


def test_SuperpopVector_rejects_missing_key() -> None:
    with pytest.raises(TypeError):
        SuperpopVector(  # type: ignore
            AFR=prob_int,
            AMR=prob_int,
            EAS=prob_int,
            EUR=prob_int,
        )


def test_SuperpopVector_extra_key() -> None:
    with pytest.raises(TypeError):
        SuperpopVector(  # type: ignore
            AFR=prob_int,
            AMR=prob_int,
            EAS=prob_int,
            EUR=prob_int,
            SAS=prob_int,
            FOO=prob_int,
        )


def test_AncestryScoresOneSample_accepts_valid_args() -> None:
    ancestry_result = AncestryScoresOneSample(
        sample_id="my_sample_id",
        top_hit=AncestryTopHit(probability=0.6, populations=["SAS"]),
        populations=PopulationVector(**pop_kwargs),
        superpops=SuperpopVector(**superpop_kwargs),
        n_snps=10,
    )
    assert type(ancestry_result.n_snps) is int


def test_AncestryScoresOneSample_rejects_invalid_n_snps() -> None:
    with pytest.raises(TypeError, match="n_snps must be non-negative"):
        AncestryScoresOneSample(
            sample_id="my_sample_id",
            top_hit=AncestryTopHit(probability=0.6, populations=["SAS"]),
            populations=PopulationVector(**pop_kwargs),
            superpops=SuperpopVector(**superpop_kwargs),
            n_snps=-10,
        )

    with pytest.raises(TypeError, match="n_snps must be an int, not <class 'float'>"):
        AncestryScoresOneSample(
            sample_id="my_sample_id",
            top_hit=AncestryTopHit(probability=0.6, populations=["SAS"]),
            populations=PopulationVector(**pop_kwargs),
            superpops=SuperpopVector(**superpop_kwargs),
            n_snps=float(10), # type: ignore
        )


def test_AncestryResults_accepts_valid_args() -> None:
    ancestry_response = AncestryResults(
        results=[
            AncestryScoresOneSample(
                sample_id="foo",
                top_hit=AncestryTopHit(probability=0.6, populations=["EAS"]),
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                n_snps=5
            ),
            AncestryScoresOneSample(
                sample_id="bar",
                top_hit=AncestryTopHit(probability=0.7, populations=["EUR"]),
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                n_snps=5
            ),
            AncestryScoresOneSample(
                sample_id="baz",
                top_hit=AncestryTopHit(probability=0.5, populations=["AFR", "AMR"]),
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                n_snps=5
            ),
        ],
        pcs={"SampleID1": [0.1, 0.2, 0.3], "SampleID2": [0.4, 0.5, 0.6], "SampleID3": [0.7, 0.8, 0.9]},
    )
    ancestry_response_json = msgspec.json.encode(ancestry_response)
    msgspec.json.decode(ancestry_response_json, type=AncestryResults)


def test_AncestryResults_rejects_invalid_pcs() -> None:
    with pytest.raises(
        msgspec.ValidationError, match=re.escape("Expected `object`, got `array` - at `$.pcs`")
    ):
        ancestry_response = AncestryResults(
            results=[
                AncestryScoresOneSample(
                    sample_id="foo",
                    top_hit=AncestryTopHit(probability=0.6, populations=["EAS"]),
                    populations=PopulationVector(**pop_kwargs),
                    superpops=SuperpopVector(**superpop_kwargs),
                    n_snps=0,
                ),
            ],
            pcs=({"SampleID1": [0.1]}, {"SampleID2": [0.4]}, {"SampleID3": [0.7]}),  # type: ignore
        )
        ancestry_response_json = msgspec.json.encode(ancestry_response)
        msgspec.json.decode(ancestry_response_json, type=AncestryResults)
