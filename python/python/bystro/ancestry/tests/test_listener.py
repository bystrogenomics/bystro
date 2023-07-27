"""Test ancestry listener."""

from pathlib import Path
from unittest.mock import patch

import pytest

from bystro.ancestry.ancestry_types import AncestrySubmission
from bystro.ancestry.listener import (
    AncestryJobData,
    _check_vcf_dir_access,
    completed_msg_fn,
    handler_fn_factory,
    submit_msg_fn,
)
from bystro.ancestry.tests.test_inference import ANCESTRY_MODEL, FAKE_GENOTYPES
from bystro.beanstalkd.messages import ProgressMessage
from bystro.beanstalkd.worker import ProgressPublisher

FAKE_VCF_DIR = Path("my_fake_vcf_dir")


handler_fn = handler_fn_factory(ANCESTRY_MODEL, FAKE_VCF_DIR)


def test_handler_fn_happy_path():
    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )
    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id2", ancestry_submission=ancestry_submission
    )
    with patch("bystro.ancestry.listener._load_vcf", return_value=FAKE_GENOTYPES) as _mock:
        ancestry_response = handler_fn(publisher, ancestry_job_data)
    assert ancestry_submission.vcf_path == ancestry_response.vcf_path


def test_submit_msg_fn_happy_path():
    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=ancestry_submission
    )
    submitted_job_message = submit_msg_fn(ancestry_job_data)
    assert submitted_job_message.submissionID == ancestry_job_data.submissionID


def test_completed_msg_fn_happy_path():
    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=ancestry_submission
    )

    with patch("bystro.ancestry.listener._load_vcf", return_value=FAKE_GENOTYPES) as _mock:
        ancestry_response = handler_fn(publisher, ancestry_job_data)
    ancestry_job_complete_message = completed_msg_fn(ancestry_job_data, ancestry_response)

    assert ancestry_job_complete_message.submissionID == ancestry_job_data.submissionID
    assert ancestry_job_complete_message.results == ancestry_response


def test_completed_msg_fn_rejects_nonmatching_vcf_paths():
    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=ancestry_submission
    )

    with patch("bystro.ancestry.listener._load_vcf", return_value=FAKE_GENOTYPES) as _mock:
        _correct_but_unused_ancestry_response = handler_fn(publisher, ancestry_job_data)

    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )

    # now instantiate another ancestry response with the wrong vcf...
    wrong_ancestry_submission = AncestrySubmission("bar.vcf")
    wrong_ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id", ancestry_submission=wrong_ancestry_submission
    )

    with patch("bystro.ancestry.listener._load_vcf", return_value=FAKE_GENOTYPES) as _:
        wrong_ancestry_response = handler_fn(publisher, wrong_ancestry_job_data)
    # end instantiating another ancestry response with the wrong vcf...

    with pytest.raises(
        ValueError, match="Ancestry submission filename .* doesn't match response filename"
    ):
        _ancestry_job_complete_message = completed_msg_fn(ancestry_job_data, wrong_ancestry_response)


def test__check_vcf_dir_access():
    with pytest.raises(
        FileNotFoundError, match="will not be able to read VCFs in order to report ancestry results"
    ):
        _check_vcf_dir_access(Path("my_fake_vcf_dir"))
