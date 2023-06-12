"""Test ancestry listener."""


import pytest

from bystro.ancestry.ancestry_types import AncestrySubmission
from bystro.ancestry.listener import _handler_fn
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
    ancestry_response = await _handler_fn(publisher, ancestry_submission)
    assert ancestry_submission.vcf_path == ancestry_response.vcf_path
