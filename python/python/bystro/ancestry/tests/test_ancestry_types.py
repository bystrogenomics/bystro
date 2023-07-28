"""Test ancestry_types.py."""
import pytest
from attrs.exceptions import FrozenInstanceError

from bystro.ancestry.ancestry_types import (
    AncestryResponse,
    AncestryResult,
    AncestrySubmission,
    AttrValidationError,
    PopulationVector,
    ProbabilityInterval,
    SuperpopVector,
)
from bystro.ancestry.train import POPS, SUPERPOPS


def test_AncestrySubmission_accepts_valid_vcf_paths():
    AncestrySubmission("foo.vcf")
    AncestrySubmission("foo.vcf.gz")
    AncestrySubmission("foo.vcf.lz4")
    AncestrySubmission("foo.vcf.zstd")
    AncestrySubmission("foo.vcf.bzip2")


def test_AncestrySubmission_rejects_bad_vcf_paths():
    with pytest.raises(AttrValidationError):
        AncestrySubmission(3)
    with pytest.raises(AttrValidationError):
        AncestrySubmission("foo.txt")
    with pytest.raises(AttrValidationError):
        AncestrySubmission("foo.gz")
    with pytest.raises(AttrValidationError):
        AncestrySubmission("foo.vcf.docx")
    with pytest.raises(AttrValidationError):
        AncestrySubmission("foo.txt.gz")


def test_AncestrySubmission_is_frozen():
    ancestry_submission = AncestrySubmission("foo.vcf")
    with pytest.raises(FrozenInstanceError):
        ancestry_submission.vcf_path = "bar.vcf"


prob_int = ProbabilityInterval(lower_bound=0.0, upper_bound=1.0)


def test_ProbabilityInterval_accepts_valid_bounds() -> None:
    """Ensure we can instantiate, validate ProbabilityInterval correctly."""
    ProbabilityInterval(lower_bound=0.1, upper_bound=0.9)


def test_ProbabilityInterval_rejects_bad_bounds() -> None:
    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=0.1, upper_bound=1.1)

    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=-2, upper_bound=1.1)

    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=1, upper_bound=0)

    with pytest.raises(FrozenInstanceError):
        prob_int.lower_bound = 0.5  # type: ignore [misc]


pop_kwargs = {pop: prob_int for pop in POPS}
superpop_kwargs = {pop: prob_int for pop in SUPERPOPS}


def test_PopulationVector_accepts_valid_args() -> None:
    """Ensure we can instantiate, validate PopulationVector correctly."""
    PopulationVector(**pop_kwargs)


def test_PopulationVector_rejects_missing_key() -> None:
    pop_kwargs_with_missing_key = pop_kwargs.copy()
    del pop_kwargs_with_missing_key["ACB"]
    with pytest.raises(AttrValidationError):
        PopulationVector(**pop_kwargs_with_missing_key)


def test_PopulationVector_rejects_extra_key() -> None:
    pop_kwargs_with_extra_key = pop_kwargs.copy()
    pop_kwargs_with_extra_key["FOO"] = prob_int
    with pytest.raises(AttrValidationError):
        PopulationVector(**pop_kwargs_with_extra_key)


def test_PopulationVector_is_frozen() -> None:
    population_vector = PopulationVector(**pop_kwargs)
    with pytest.raises(FrozenInstanceError):
        population_vector.ACB = prob_int  # type: ignore [misc]


def test_SuperpopVector_accepts_valid_args() -> None:
    prob_int = ProbabilityInterval(lower_bound=0.0, upper_bound=1.0)
    SuperpopVector(
        AFR=prob_int,
        AMR=prob_int,
        EAS=prob_int,
        EUR=prob_int,
        SAS=prob_int,
    )


def test_SuperpopVector_rejects_missing_key() -> None:
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


def test_SuperpopVector_is_frozen() -> None:
    superpop_vector = SuperpopVector(
        AFR=prob_int,
        AMR=prob_int,
        EAS=prob_int,
        EUR=prob_int,
        SAS=prob_int,
    )

    with pytest.raises(FrozenInstanceError):
        superpop_vector.AFR = prob_int  # type: ignore [misc]


def test_AncestryResult_accepts_valid_args() -> None:
    AncestryResult(
        sample_id="my_sample_id",
        populations=PopulationVector(**pop_kwargs),
        superpops=SuperpopVector(**superpop_kwargs),
        missingness=0.5,
    )


def test_AncestryResult_rejects_invalid_missingness() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResult(
            sample_id="my_sample_id",
            populations=PopulationVector(**pop_kwargs),
            superpops=SuperpopVector(**superpop_kwargs),
            missingness=1.1,
        )


def test_AncestryResult_is_frozen() -> None:
    ancestry_result = AncestryResult(
        sample_id="my_sample_id",
        populations=PopulationVector(**pop_kwargs),
        superpops=SuperpopVector(**superpop_kwargs),
        missingness=0.1,
    )
    with pytest.raises(FrozenInstanceError):
        ancestry_result.missingness = 0.2  # type: ignore [misc]


def test_AncestryResponse_accepts_valid_args() -> None:
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


def test_AncestryResponse_rejects_bad_vcf_path() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResponse(vcf_path="foo.txt", results=[])


def test_AncestryResponse_rejects_bad_results_type() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResponse(
            vcf_path="myfile.vcf",
            results=[3, 4, 5],  # type: ignore [list-item]
        )


def test_AncestryResponse_is_frozen() -> None:
    ancestry_response = AncestryResponse(vcf_path="foo.vcf", results=[])
    with pytest.raises(FrozenInstanceError):
        ancestry_response.vcf_path = "bar.vcf"  # type: ignore [misc]


def test_AncestryResponse_rejects_duplicate_sample_ids() -> None:
    with pytest.raises(
        AttrValidationError, match=r"Expected unique sample_ids but found duplicated samples {'foo'}"
    ):
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
            ],
        )
