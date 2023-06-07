"""Provide a listener to allow the ancestry model to talk over beanstalk."""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from sample_response import sample_ancestry_response

from ancestry.ancestry_types import AncestryResponse, AncestrySubmission
from ancestry.beanstalk import (
    BeanstalkEvent,
    BeanstalkEventMessage,
    BeanstalkSubmissionMessage,
    BystroBeanstalkClient,
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BEANSTALK_TIMEOUT_ERROR = "TIMED_OUT"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file_handler:
        return YAML(typ="safe").load(file_handler)


def _ancestry_response_from_ancestry_submission(
    ancestry_submission: AncestrySubmission,
) -> AncestryResponse:
    response = sample_ancestry_response.copy()
    response.vcf_path = ancestry_submission.vcf_path
    return response


def _make_event_from_submission(sub_msg: BeanstalkSubmissionMessage) -> BeanstalkEventMessage:
    ancestry_sub = AncestrySubmission(**sub_msg.data)
    ancestry_response = _ancestry_response_from_ancestry_submission(ancestry_sub)
    queue_id = str(int(sub_msg.queue_id) + 1)
    submission_id = str(int(sub_msg.submission_id) + 1)
    return BeanstalkEventMessage(
        event=BeanstalkEvent.COMPLETED,
        queue_id=queue_id,
        submission_id=submission_id,
        data=ancestry_response,
    )


def main() -> None:
    """Run ancestry server accepting genotype requests and rendering global ancestry predictions."""
    parser = argparse.ArgumentParser(description="Run the ancestry server.")
    parser.add_argument(
        "--queue_conf",
        type=str,
        required=True,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
    )
    args = parser.parse_args()
    beanstalk_client = BystroBeanstalkClient.from_config(args.queue_conf, worker_name="ancestry")

    while True:
        beanstalk_client.handle_job(_make_event_from_submission)


if __name__ == "__main__":
    main()
