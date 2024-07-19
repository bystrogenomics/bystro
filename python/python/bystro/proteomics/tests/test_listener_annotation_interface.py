from msgspec import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from bystro.proteomics.listener_annotation_interface import (
    ProteomicsJobData,
    make_handler_fn,
    submit_msg_fn,
    completed_msg_fn,
    SubmittedJobMessage,
    ProteomicsJobCompleteMessage,
)

from bystro.beanstalkd.worker import ProgressPublisher

pd.options.future.infer_string = True  # type: ignore

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
    mocker.patch(
        "bystro.proteomics.listener_annotation_interface.ds.dataset",
        return_value=mocker.Mock(
            to_table=mocker.Mock(to_pandas=mocker.Mock(return_value=FAKE_PROTEOMICS_DATA))
        ),
    )
    mocker.patch(
        "bystro.proteomics.listener_annotation_interface.pd.read_csv", return_value=FAKE_PROTEOMICS_DATA
    )
    mocker.patch(
        "bystro.proteomics.listener_annotation_interface.get_annotation_result_from_query",
        return_value=pd.DataFrame(),
    )
    mocker.patch(
        "bystro.proteomics.listener_annotation_interface.join_annotation_result_to_proteomic_dataset",
        return_value=pd.DataFrame(),
    )
    mocker.patch("bystro.proteomics.annotation_interface.AsyncOpenSearch", return_value=mocker.Mock())

    job_data = ProteomicsJobData(
        data_path=str(tmpdir / "some_data.feather"),
        out_dir=str(tmpdir),
        annotation_query=FAKE_ANNOTATION_QUERY,
        index_name=FAKE_INDEX_NAME,
        submission_id="test_submission",
    )

    publisher = mocker.Mock(spec=ProgressPublisher)

    handler_fn = make_handler_fn({})
    result = handler_fn(publisher, job_data)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    expected_path = str(Path(job_data.out_dir) / f"joined.{timestamp}.feather")

    assert result == expected_path


def test_completion_fn(tmpdir):
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id",
        data_path="some_data.feather",
        out_dir=str(tmpdir),
        annotation_query=FAKE_ANNOTATION_QUERY,
        index_name=FAKE_INDEX_NAME,
    )

    timestamp = "20230315_120000+0000"
    result_path = tmpdir / f"joined.{timestamp}.feather"

    completed_msg = completed_msg_fn(proteomics_job_data, str(result_path))

    assert isinstance(completed_msg, ProteomicsJobCompleteMessage)
    assert completed_msg.result_path == str(result_path)


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
