from pathlib import Path

from msgspec import json
import pytest

from bystro.ancestry.listener import (
    AncestryJobData,
    completed_msg_fn,
    AncestryJobCompleteMessage,
    AncestryResults,
)
from bystro.ancestry.inference import AncestryModels
from bystro.ancestry.tests.test_inference import (
    ANCESTRY_MODEL,
    _infer_ancestry,
)
from bystro.api.ancestry import calculate_ancestry_scores


@pytest.mark.integration("requires bystro-vcf to be installed")
def test_calculate_ancestry_scores_happy_path(mocker, tmpdir):
    mocker.patch(
        "bystro.ancestry.model.get_models_from_s3",
        return_value=AncestryModels(ANCESTRY_MODEL, ANCESTRY_MODEL),
    )

    VCF_PATH = Path(__file__).parent / "trio.trim.vep.vcf.gz"
    ancestry_response = calculate_ancestry_scores(
        str(VCF_PATH), "hg19", dosage=False, out_dir=str(tmpdir)
    )

    assert isinstance(ancestry_response, AncestryResults)

    # Demonstrate that all expected sample_ids are accounted for
    samples_seen = set()
    expected_samples = set(["1805", "4805", "1847"])
    for result in ancestry_response.results:
        samples_seen.add(result.sample_id)

    assert samples_seen == expected_samples


def test_completion_fn(tmpdir):
    ancestry_job_data = AncestryJobData(
        submission_id="my_submission_id2",
        dosage_matrix_path="some_dosage.feather",
        out_dir=str(tmpdir),
        assembly="hg38",
    )

    ancestry_results, _ = _infer_ancestry()

    completed_msg = completed_msg_fn(ancestry_job_data, ancestry_results)

    assert isinstance(completed_msg, AncestryJobCompleteMessage)


def test_completion_message():
    ancestry_job_data = AncestryJobCompleteMessage(
        submission_id="my_submission_id2", result_path="some_dosage.feather"
    )

    serialized_values = json.encode(ancestry_job_data)
    expected_value = {
        "submissionId": "my_submission_id2",
        "event": "completed",
        "resultPath": "some_dosage.feather",
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=AncestryJobCompleteMessage)
    assert deserialized_values == ancestry_job_data


def test_job_data_from_beanstalkd():
    ancestry_job_data = AncestryJobData(
        submission_id="my_submission_id2",
        dosage_matrix_path="some_dosage.feather",
        out_dir="/foo",
        assembly="hg38",
    )

    serialized_values = json.encode(ancestry_job_data)
    expected_value = {
        "submissionId": "my_submission_id2",
        "dosageMatrixPath": "some_dosage.feather",
        "outDir": "/foo",
        "assembly": "hg38",
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=AncestryJobData)
    assert deserialized_values == ancestry_job_data
