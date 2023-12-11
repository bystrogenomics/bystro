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


# ruff: noqa: E721

# In several tests we explicitly check that a value `is float` rather
# than use the more pythonic `isinstance(value, float)`.  We make
# these explicit checks in order to ensure that such values are raw
# floats and not np.float64's, which can't easily be deserialized.
# But the former method of checking raises E71 errors, which are
# exempted from the linter on a file-wide basis above.


def test_AncestrySubmission_accepts_valid_vcf_paths():
    AncestrySubmission("foo.vcf")
    AncestrySubmission("foo.vcf.gz")
    AncestrySubmission("foo.vcf.lz4")
    AncestrySubmission("foo.vcf.zstd")
    AncestrySubmission("foo.vcf.bzip2")


def test_AncestrySubmission_rejects_bad_vcf_paths():
    with pytest.raises(AttrValidationError):
        AncestrySubmission(3)  # type: ignore
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
        ancestry_submission.vcf_path = "bar.vcf"  # type: ignore


prob_int = ProbabilityInterval(lower_bound=0.0, upper_bound=1.0)


def test_ProbabilityInterval_accepts_valid_bounds() -> None:
    """Ensure we can instantiate, validate ProbabilityInterval correctly."""
    prob_int = ProbabilityInterval(lower_bound=0.1, upper_bound=0.9)
    assert type(prob_int.lower_bound) is float
    assert type(prob_int.upper_bound) is float


def test_ProbabilityInterval_rejects_bad_bounds() -> None:
    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=0.1, upper_bound=1.1)

    with pytest.raises(AttrValidationError):
        ProbabilityInterval(lower_bound=-2, upper_bound=1.1)

    err_msg = (
        "Lower bound must be less than or equal to upper bound.  "
        "Got: lower_bound=1.0, upper_bound=0.0 instead."
    )
    with pytest.raises(
        AttrValidationError,
        match=err_msg,
    ):
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
    ancestry_result = AncestryResult(
        sample_id="my_sample_id",
        top_hit=(0.6, ["SAS"]),
        populations=PopulationVector(**pop_kwargs),
        superpops=SuperpopVector(**superpop_kwargs),
        missingness=0.5,
    )
    assert type(ancestry_result.missingness) is float


def test_AncestryResult_rejects_invalid_missingness() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResult(
            sample_id="my_sample_id",
            top_hit=(0.6, ["SAS"]),
            populations=PopulationVector(**pop_kwargs),
            superpops=SuperpopVector(**superpop_kwargs),
            missingness=1.1,
        )


def test_AncestryResult_is_frozen() -> None:
    ancestry_result = AncestryResult(
        sample_id="my_sample_id",
        top_hit=(0.6, ["SAS"]),
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
                top_hit=(0.6, ["EAS"]),
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                missingness=0.5,
            ),
            AncestryResult(
                sample_id="bar",
                top_hit=(0.7, ["EUR"]),
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                missingness=0.5,
            ),
            AncestryResult(
                sample_id="baz",
                top_hit=(0.5, ["AFR", "AMR"]),
                populations=PopulationVector(**pop_kwargs),
                superpops=SuperpopVector(**superpop_kwargs),
                missingness=0.5,
            ),
        ],
        pcs={"SampleID1": [0.1, 0.2, 0.3], "SampleID2": [0.4, 0.5, 0.6], "SampleID3": [0.7, 0.8, 0.9]},
    )


def test_AncestryResponse_rejects_bad_vcf_path() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResponse(
            vcf_path="foo.txt",
            results=[],
            pcs={
                "SampleID1": [0.1, 0.2, 0.3],
                "SampleID2": [0.4, 0.5, 0.6],
                "SampleID3": [0.7, 0.8, 0.9],
            },
        )


def test_AncestryResponse_rejects_bad_results_type() -> None:
    with pytest.raises(AttrValidationError):
        AncestryResponse(
            vcf_path="myfile.vcf",
            results=[3, 4, 5],  # type: ignore [list-item]
            pcs={
                "SampleID1": [0.1, 0.2, 0.3],
                "SampleID2": [0.4, 0.5, 0.6],
                "SampleID3": [0.7, 0.8, 0.9],
            },
        )


def test_AncestryResponse_is_frozen() -> None:
    ancestry_response = AncestryResponse(vcf_path="foo.vcf", results=[], pcs={})
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
                    top_hit=(0.6, ["EAS"]),
                    populations=PopulationVector(**pop_kwargs),
                    superpops=SuperpopVector(**superpop_kwargs),
                    missingness=0.5,
                ),
                AncestryResult(
                    sample_id="foo",
                    top_hit=(0.6, ["EAS"]),
                    populations=PopulationVector(**pop_kwargs),
                    superpops=SuperpopVector(**superpop_kwargs),
                    missingness=0.5,
                ),
                AncestryResult(
                    sample_id="bar",
                    top_hit=(0.7, ["EUR"]),
                    populations=PopulationVector(**pop_kwargs),
                    superpops=SuperpopVector(**superpop_kwargs),
                    missingness=0.5,
                ),
            ],
            pcs={
                "SampleID1": [0.1, 0.2, 0.3],
                "SampleID2": [0.4, 0.5, 0.6],
                "SampleID3": [0.7, 0.8, 0.9],
            },
        )
