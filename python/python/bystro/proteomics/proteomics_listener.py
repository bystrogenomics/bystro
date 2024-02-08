"""Listener for proteomics worker."""
import argparse
import logging
from pathlib import Path

from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage
from bystro.beanstalkd.worker import ProgressPublisher, QueueConf, get_progress_reporter, listen
from bystro.proteomics.proteomics import load_fragpipe_dataset
from bystro.proteomics.proteomics_types import (
    ProteomicsResponse,
    ProteomicsSubmission,
    DataFrameJson,
)
from ruamel.yaml import YAML

logging.basicConfig(filename="proteomics_listener.log", level=logging.INFO)
logger = logging.getLogger()

PROTEOMICS_TUBE = "proteomics"


class ProteomicsJobData(BaseMessage, frozen=True):
    """Wrap an ProteomicsSubmission in a BaseMessage for beanstalk."""

    proteomics_submission: ProteomicsSubmission


class ProteomicsJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    """Wrap an ProteomicsResponse in a CompletedJobMessage for beanstalk."""

    results: ProteomicsResponse


def submit_msg_fn(proteomics_job_data: ProteomicsJobData) -> SubmittedJobMessage:
    """Acknowledge receipt of ProteomicsJobData."""
    logger.debug("entering submit_msg_fn: %s", proteomics_job_data)
    return SubmittedJobMessage(proteomics_job_data.submission_id)


def handler_fn(
    progress_publisher: ProgressPublisher, proteomics_job_data: ProteomicsJobData
) -> ProteomicsResponse:
    """Take a filename wrapped in ProteomicsJobData and return a df wrapped in ProteomicsResponse."""
    _reporter = get_progress_reporter(progress_publisher)
    tsv_filename = proteomics_job_data.proteomics_submission.tsv_filename
    fragpipe_df = load_fragpipe_dataset(tsv_filename)
    return ProteomicsResponse(tsv_filename, DataFrameJson.from_df(fragpipe_df))


def completed_msg_fn(
    proteomics_job_data: ProteomicsJobData, proteomics_response: ProteomicsResponse
) -> ProteomicsJobCompleteMessage:
    """Send job complete message."""
    logger.debug("entering completed_msg_fn: %s", proteomics_job_data)
    proteomics_submission = proteomics_job_data.proteomics_submission
    if proteomics_submission.tsv_filename != proteomics_response.tsv_filename:
        err_msg = (
            f"Proteomics submission filename {proteomics_submission.tsv_filename} "
            f"doesn't match response filename {proteomics_response.tsv_filename}: this is a bug."
        )
        raise ValueError(err_msg)
    return ProteomicsJobCompleteMessage(
        submission_id=proteomics_job_data.submission_id, results=proteomics_response
    )


# todo(pat): refactor this into bystro.beanstalkd
def _load_queue_conf(queue_conf_path: str) -> QueueConf:
    with Path(queue_conf_path).open(encoding="utf-8") as queue_config_file:
        raw_queue_conf = YAML(typ="safe").load(queue_config_file)
    beanstalk_conf = raw_queue_conf["beanstalkd"]
    return QueueConf(addresses=beanstalk_conf["addresses"], tubes=beanstalk_conf["tubes"])


def main(queue_conf: QueueConf) -> None:
    """Run proteomics listener."""
    logger.info(
        "Proteomics worker is listening on addresses: %s, tube: %s...",
        queue_conf.addresses,
        PROTEOMICS_TUBE,
    )

    listen(
        ProteomicsJobData,
        handler_fn,
        submit_msg_fn,
        completed_msg_fn,
        queue_conf,
        PROTEOMICS_TUBE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some config files.")
    parser.add_argument(
        "--queue_conf",
        type=Path,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
        required=True,
    )
    args = parser.parse_args()
    queue_conf = _load_queue_conf(args.queue_conf)

    main(queue_conf)
