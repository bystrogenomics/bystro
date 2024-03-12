from msgspec import json
import pyarrow.feather as feather  # type: ignore
import pandas as pd

from bystro.proteomics.listener_annotation_interface import (
    ProteomicsJobData,
    handler_fn,
    submit_msg_fn,
    completed_msg_fn,
    SubmittedJobMessage,
    ProteomicsJobCompleteMessage,
)

from bystro.beanstalkd.messages import ProgressMessage
from bystro.beanstalkd.worker import ProgressPublisher


FAKE_PROTEOMICS_DATA = pd.DataFrame(
    {"sample_id": ["sample1", "sample2"], "gene_name": ["gene1", "gene2"], "abundance": [100, 200]}
)

FAKE_ANNOTATION_QUERY = "gene_name:TP53"
FAKE_INDEX_NAME = "mock_index"


def test_submit_fn():
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id",
        data_path="some_data.feather",
        out_dir="/path/to/some/dir",
        annotation_query=FAKE_ANNOTATION_QUERY,
        index_name=FAKE_INDEX_NAME,
    )
    submitted_job_message = submit_msg_fn(proteomics_job_data)

    assert isinstance(submitted_job_message, SubmittedJobMessage)


def test_handler_fn_happy_path(tmpdir, mocker):
    data_path = "some_data.feather"
    f1 = tmpdir.join(data_path)

    feather.write_feather(FAKE_PROTEOMICS_DATA, str(f1))

    progress_message = ProgressMessage(submission_id="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    mocker.patch(
        "bystro.proteomics.listener_annotation_interface.get_annotation_result_from_query",
        return_value=pd.DataFrame(),
    )
    mocker.patch(
        "bystro.proteomics.listener_annotation_interface.join_annotation_result_to_proteomics_dataset",
        return_value=pd.DataFrame(),
    )

    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id",
        data_path=str(f1),
        out_dir=str(tmpdir),
        annotation_query=FAKE_ANNOTATION_QUERY,
        index_name=FAKE_INDEX_NAME,
    )

    result_path = handler_fn(publisher, proteomics_job_data)

    assert result_path == str(tmpdir / "joined_results.feather")


def test_completion_fn(tmpdir):
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id",
        data_path="some_data.feather",
        out_dir=str(tmpdir),
        annotation_query=FAKE_ANNOTATION_QUERY,
        index_name=FAKE_INDEX_NAME,
    )

    completed_msg = completed_msg_fn(proteomics_job_data, str(tmpdir / "joined_results.feather"))

    assert isinstance(completed_msg, ProteomicsJobCompleteMessage)
    assert completed_msg.result_path == str(tmpdir / "joined_results.feather")


def test_job_data_from_beanstalkd():
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id",
        data_path="some_data.feather",
        out_dir="/foo",
        annotation_query=FAKE_ANNOTATION_QUERY,
        index_name=FAKE_INDEX_NAME,
    )

    serialized_values = json.encode(proteomics_job_data)
    expected_value = {
        "submissionId": "my_submission_id",
        "dataPath": "some_data.feather",
        "outDir": "/foo",
        "annotationQuery": FAKE_ANNOTATION_QUERY,
        "indexName": FAKE_INDEX_NAME,
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=ProteomicsJobData)
    assert deserialized_values == proteomics_job_data
