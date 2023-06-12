"""Test ancestry listener."""


import pytest

from bystro.ancestry.ancestry_types import AncestrySubmission
from bystro.ancestry.listener import AncestryJobData, _handler_fn
from bystro.beanstalkd.messages import ProgressMessage
from bystro.beanstalkd.worker import ProgressPublisher

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio()
async def test__handler_fn():
    progress_message = ProgressMessage(submissionID="my_submission_id")
    publisher = ProgressPublisher(
        host="127.0.0.1", port=1234, queue="my_queue", message=progress_message
    )
    ancestry_submission = AncestrySubmission("foo.vcf")
    ancestry_job_data = AncestryJobData(
        submissionID="my_submission_id2", ancestry_submission=ancestry_submission
    )
    ancestry_response = await _handler_fn(publisher, ancestry_job_data)
    assert ancestry_submission.vcf_path == ancestry_response.vcf_path
