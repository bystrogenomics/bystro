import pytest
from bystro.proteomics.proteomics_listener import (
    submit_msg_fn,
    ProteomicsJobData,
    handler_fn,
    completed_msg_fn,
)
from bystro.proteomics.proteomics_types import ProteomicsSubmission
from bystro.beanstalkd.messages import ProgressMessage
from bystro.beanstalkd.worker import ProgressPublisher
import pandas as pd
from unittest.mock import patch

FAKE_FRAGPIPE_DF = pd.DataFrame(
    {
        "ACADVL": {"Sample1": 0.1, "Sample2": 0.2},
        "ACAT1": {"Sample1": 0.1, "Sample2": 0.2},
        "ACVRL1": {"Sample1": 0.1, "Sample2": 0.2},
    }
)

#  The namespace of load_fragpipe_dataset may be surprising here, but we need to patch this
#  according to its (imported) location in the module under test, not where it's originally defined.
LOAD_FRAGPIPE_DATASET_PATCH_TARGET = "bystro.proteomics.proteomics_listener.load_fragpipe_dataset"


def test_submit_msg_fn_happy_path():
    proteomics_submission = ProteomicsSubmission("foo.tsv")
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id", proteomics_submission=proteomics_submission
    )
    submitted_job_message = submit_msg_fn(proteomics_job_data)
    assert proteomics_job_data.submission_id == submitted_job_message.submission_id


def test_handler_fn_happy_path():
    progress_message = ProgressMessage(submission_id="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )
    proteomics_submission = ProteomicsSubmission("foo.tsv")
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id2", proteomics_submission=proteomics_submission
    )
    with patch(LOAD_FRAGPIPE_DATASET_PATCH_TARGET, return_value=FAKE_FRAGPIPE_DF) as _mock:
        proteomics_response = handler_fn(publisher, proteomics_job_data)
    assert proteomics_submission.tsv_filename == proteomics_response.tsv_filename


def test_completed_msg_fn_happy_path():
    progress_message = ProgressMessage(submission_id="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    proteomics_submission = ProteomicsSubmission("foo.tsv")
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id", proteomics_submission=proteomics_submission
    )

    with patch(LOAD_FRAGPIPE_DATASET_PATCH_TARGET, return_value=FAKE_FRAGPIPE_DF) as _mock:
        proteomics_response = handler_fn(publisher, proteomics_job_data)
    proteomics_job_complete_message = completed_msg_fn(proteomics_job_data, proteomics_response)

    assert proteomics_job_complete_message.submission_id == proteomics_job_data.submission_id
    assert proteomics_job_complete_message.results == proteomics_response


def test_completed_msg_fn_filenames_dont_match():
    progress_message = ProgressMessage(submission_id="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    proteomics_submission = ProteomicsSubmission("foo.tsv")
    proteomics_job_data = ProteomicsJobData(
        submission_id="my_submission_id", proteomics_submission=proteomics_submission
    )
    wrong_proteomics_submission = ProteomicsSubmission("wrong_file.tsv")
    wrong_proteomics_job_data = ProteomicsJobData(
        submission_id="wrong_submission_id", proteomics_submission=wrong_proteomics_submission
    )

    with patch(LOAD_FRAGPIPE_DATASET_PATCH_TARGET, return_value=FAKE_FRAGPIPE_DF) as _mock:
        proteomics_response = handler_fn(publisher, proteomics_job_data)

    with pytest.raises(ValueError, match=r".*\.tsv doesn't match response filename .*\.tsv"):
        _proteomics_job_complete_message = completed_msg_fn(
            wrong_proteomics_job_data, proteomics_response
        )
