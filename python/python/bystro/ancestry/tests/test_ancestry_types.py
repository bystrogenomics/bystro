"""Test ancestry_types.py."""
import pytest
from pydantic import ValidationError

from ..ancestry_types import (
    AncestrySubmission,
    AttrValidationError,
    AncestryResponse,
    AncestryResult,
    PopulationVector,
    ProbabilityInterval,
    SuperpopVector,
)

POPULATIONS = [
    "ACB",
    "ASW",
    "BEB",
    "CDX",
    "CEU",
    "CHB",
    "CHS",
    "CLM",
    "ESN",
    "FIN",
    "GBR",
    "GIH",
    "GWD",
    "IBS",
    "ITU",
    "JPT",
    "KHV",
    "LWK",
    "MAG",
    "MSL",
    "MXL",
    "PEL",
    "PJL",
    "PUR",
    "STU",
    "TSI",
    "YRI",
]

SUPERPOPS = [
    "AFR",
    "AMR",
    "EAS",
    "EUR",
    "SAS",
]


def test_AncestrySubmission():
    AncestrySubmission("foo.vcf")


def test_AncestrySubmission_bad_filepath():
    with pytest.raises(AttrValidationError):
        AncestrySubmission("foo.txt")


def test_ProbabilityInterval() -> None:
    """Ensure we can instantiate, validate ProbabilityInterval correctly."""
    ProbabilityInterval(lower_bound=0.1, upper_bound=0.9)

    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=0.1, upper_bound=1.1)

    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=-2, upper_bound=1.1)

    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=1, upper_bound=0)


prob_int = ProbabilityInterval(lower_bound=0.0, upper_bound=1.0)
pop_kwargs = {pop: prob_int for pop in POPULATIONS}
superpop_kwargs = {pop: prob_int for pop in SUPERPOPS}


def test_PopulationVector() -> None:
    """Ensure we can instantiate, validate PopulationVector correctly."""
    PopulationVector(**pop_kwargs)


def test_PopulationVector_with_missing_key() -> None:
    pop_kwargs_with_missing_key = pop_kwargs.copy()
    del pop_kwargs_with_missing_key["ACB"]
    with pytest.raises(AttrValidationError):
        PopulationVector(**pop_kwargs_with_missing_key)


def test_PopulationVector_with_extra_key() -> None:
    pop_kwargs_with_extra_key = pop_kwargs.copy()
    pop_kwargs_with_extra_key["FOO"] = prob_int
    with pytest.raises(AttrValidationError):
        PopulationVector(**pop_kwargs_with_extra_key)


def test_SuperpopVector() -> None:
    prob_int = ProbabilityInterval(lower_bound=0.0, upper_bound=1.0)
    SuperpopVector(
        AFR=prob_int,
        AMR=prob_int,
        EAS=prob_int,
        EUR=prob_int,
        SAS=prob_int,
    )


def test_SuperpopVector_missing_key() -> None:
    with pytest.raises(AttrValidationError):
        SuperpopVector(  # type: ignore [call-arg]
            AFR=prob_int,
            AMR=prob_int,
            EAS=prob_int,
            EUR=prob_int,
        )


def test_SuperpopVector_extra_key() -> None:
    with pytest.raises(AttrValidationError):
        SuperpopVector(  # type: ignore [call-arg]
            AFR=prob_int,
            AMR=prob_int,
            EAS=prob_int,
            EUR=prob_int,
            SAS=prob_int,
            FOO=prob_int,
        )


def test_AncestryResult() -> None:
    AncestryResult(
        sample_id="my_sample_id",
        populations=PopulationVector(**pop_kwargs),
        superpops=SuperpopVector(**superpop_kwargs),
        missingness=0.5,
    )


def test_AncestryResult_invalid_missingness() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResult(
            sample_id="my_sample_id",
            populations=PopulationVector(**pop_kwargs),
            superpops=SuperpopVector(**superpop_kwargs),
            missingness=1.1,
        )


def test_AncestryResponse() -> None:
    AncestryResponse(
        vcf_path="myfile.vcf",
        results=[
            AncestryResult(
                sample_id="foo",
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                missingness=0.5,
            ),
            AncestryResult(
                sample_id="bar",
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                missingness=0.5,
            ),
            AncestryResult(
                sample_id="baz",
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                missingness=0.5,
            ),
        ],
    )


def test_AncestryResponse_bad_inputs() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResponse("foo.txt", results=[])
    with pytest.raises(AttrValidationError):
        AncestryResponse(
            vcf_path="myfile.vcf",
            results=[3, 4, 5],
        )
