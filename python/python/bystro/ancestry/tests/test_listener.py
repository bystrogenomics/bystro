from msgspec import json
import pyarrow.feather as feather  # type: ignore

from bystro.ancestry.listener import (
    AncestryJobData,
    handler_fn_factory,
    submit_msg_fn,
    completed_msg_fn,
    SubmittedJobMessage,
    AncestryJobCompleteMessage,
    AncestryResults,
)
from bystro.ancestry.tests.test_inference import (
    ANCESTRY_MODEL,
    FAKE_GENOTYPES,
    FAKE_GENOTYPES_DOSAGE_MATRIX,
    _infer_ancestry,
)
from bystro.beanstalkd.messages import ProgressMessage
from bystro.beanstalkd.worker import ProgressPublisher


handler_fn = handler_fn_factory(ANCESTRY_MODEL)


def test_submit_fn():
    ancestry_job_data = AncestryJobData(
        submission_id="my_submission_id2",
        dosage_matrix_path="some_dosage.feather",
        out_dir="/path/to/some/dir",
    )
    submitted_job_message = submit_msg_fn(ancestry_job_data)

    assert isinstance(submitted_job_message, SubmittedJobMessage)


def test_handler_fn_happy_path(tmpdir):
    dosage_path = "some_dosage.feather"
    f1 = tmpdir.join(dosage_path)

    feather.write_feather(FAKE_GENOTYPES_DOSAGE_MATRIX.to_table(), str(f1))

    progress_message = ProgressMessage(submission_id="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )
    ancestry_job_data = AncestryJobData(
        submission_id="my_submission_id2", dosage_matrix_path=f1, out_dir=str(tmpdir)
    )
    ancestry_response = handler_fn(publisher, ancestry_job_data)

    assert isinstance(ancestry_response, AncestryResults)

    # Demonstrate that all expected sample_ids are accounted for
    samples_seen = set()
    expected_samples = set(FAKE_GENOTYPES.columns)
    for result in ancestry_response.results:
        samples_seen.add(result.sample_id)

    assert samples_seen == expected_samples


def test_completion_fn(tmpdir):
    ancestry_job_data = AncestryJobData(
        submission_id="my_submission_id2", dosage_matrix_path="some_dosage.feather", out_dir=str(tmpdir)
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
        submission_id="my_submission_id2", dosage_matrix_path="some_dosage.feather", out_dir="/foo"
    )

    serialized_values = json.encode(ancestry_job_data)
    expected_value = {
        "submissionId": "my_submission_id2",
        "dosageMatrixPath": "some_dosage.feather",
        "outDir": "/foo",
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=AncestryJobData)
    assert deserialized_values == ancestry_job_data
