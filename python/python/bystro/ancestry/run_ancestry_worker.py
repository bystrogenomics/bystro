"""Provide a listener to allow the ancestry model to talk over beanstalk."""
import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Any

from pystalk import BeanstalkClient, BeanstalkError
from ruamel.yaml import YAML
from bystro.ancestry.sample_response import sample_ancestry_response

from bystro.ancestry.ancestry_types import AncestryResponse, AncestrySubmission
from bystro.ancestry.beanstalk import (
    Address,
    BeanstalkEvent,
    BeanstalkEventMessage,
    BeanstalkSubmissionMessage,
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


# def _execute_job(job_data: str) -> str:
#     """Represent dummy job that just extracts the vcf for now."""


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
    beanstalk_conf = _load_yaml(Path(args.queue_conf))["beanstalkd"]
    addresses: dict[str, Any] = beanstalk_conf["addresses"]
    ancestry_tubes: dict[str, Any] = beanstalk_conf["tubes"]["ancestry"]
    submission_tube = ancestry_tubes["submission"]
    events_tube = ancestry_tubes["events"]

    # todo: refactor multiple client logic
    beanstalk_clients = []
    for address in addresses:
        parsed_address = Address.from_str(address)
        client = BeanstalkClient(parsed_address.host, parsed_address.port)
        client.watchlist = {submission_tube}
        client.use(events_tube)
        logger.info(
            "starting beanstalk client on: %s using tube(s): %s", parsed_address, client.watchlist
        )
        beanstalk_clients.append(client)
    num_clients = len(beanstalk_clients)

    for client_idx in itertools.count():
        logger.debug("starting ancestry listening loop with %s", client_idx)
        client = beanstalk_clients[client_idx % num_clients]
        try:
            job = client.reserve_job()
            logger.debug("reserved job %s", job)
        except BeanstalkError as err:
            if err.message == BEANSTALK_TIMEOUT_ERROR:
                logger.debug("Timed out while reserving a job: this is expected if no jobs are present")
                continue
            raise
        try:
            logger.debug("trying to handle a job")
            decoded_job_data = job.job_data.decode()
            logger.debug("decoded_job_data %s", decoded_job_data)
            beanstalk_msg = BeanstalkSubmissionMessage.parse_raw(decoded_job_data)
            logger.debug("beanstalk message: %s", beanstalk_msg)
            ancestry_submission = AncestrySubmission(**beanstalk_msg.data)
            logger.debug("ancestry submission: %s", ancestry_submission)
            ancestry_response = _ancestry_response_from_ancestry_submission(ancestry_submission)
            beanstalk_event_message = BeanstalkEventMessage(
                event=BeanstalkEvent.COMPLETED,
                queue_id="123",
                submission_id="456",
                data=ancestry_response,
            )
            client.put_job(beanstalk_event_message.json())
            logger.debug("successfully put job")
        except Exception:
            logger.exception("Encountered exception while handling job %s", job)
            client.release_job(job.job_id)
            raise
        client.delete_job(job.job_id)


if __name__ == "__main__":
    main()
