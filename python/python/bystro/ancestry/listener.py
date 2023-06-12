"""Provide a worker for the ancestry model."""
from bystro.beanstalkd.messages import BaseMessage, SubmittedJobMessage, CompletedJobMessage, Struct
from bystro.beanstalkd.worker import listen, QueueConf
from bystro.beanstalkd.worker import ProgressPublisher
from ruamel.yaml import YAML

from bystro.ancestry.ancestry_types import AncestrySubmission, AncestryResponse
import argparse
import os

# handler_fn

# submit_msg_fn


class AncestryJobData(BaseMessage, frozen=True):
    """Data for SaveFromQuery jobs received from beanstalkd"""

    ancestry_submission: AncestrySubmission


ANCESTRY_TUBE = "ancestry"


class AncestryJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    results: AncestryResponse


# completed_msg_fn
def _load_queue_conf(queue_conf_path: str) -> QueueConf:
    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        raw_queue_conf = YAML(typ="safe").load(queue_config_file)
    beanstalk_conf = raw_queue_conf["beanstalkd"]
    return QueueConf(addresses=beanstalk_conf["addresses"], tubes=beanstalk_conf["tubes"])


async def _go(ancestry_job_data: AncestryJobData) -> AncestryResponse:
    """Main ancestry job runner."""
    print("got ancestry job_data in go:", ancestry_job_data)
    vcf_path = ancestry_job_data.ancestry_submission.vcf_path
    return AncestryResponse(vcf_path=vcf_path, results=[])


def _handler_fn(
    publisher: ProgressPublisher, ancestry_submission: AncestrySubmission
) -> AncestryResponse:
    print("got ancestry job:", ancestry_submission)
    return _go(ancestry_submission)


def submit_msg_fn(ancestry_submission: AncestrySubmission):
    return SubmittedJobMessage(ancestry_submission.submissionID)


def completed_msg_fn(ancestry_submission: AncestrySubmission, ancestry_response: AncestryResponse):
    print("completed message:", ancestry_submission, ancestry_response)
    return AncestryJobCompleteMessage(
        submissionID=ancestry_submission.submissionID, results=ancestry_response
    )  # noqa: E501


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some config files.")
    parser.add_argument(
        "--queue_conf",
        type=str,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
        required=True,
    )
    args = parser.parse_args()

    queue_conf = _load_queue_conf(args.queue_conf)

    listen(
        AncestryJobData,
        _handler_fn,
        submit_msg_fn,
        completed_msg_fn,
        queue_conf,
        ANCESTRY_TUBE,
    )
