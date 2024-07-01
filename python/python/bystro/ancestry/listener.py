"""Provide a worker for the ancestry model."""

import argparse
import logging
from pathlib import Path

import msgspec
import pyarrow.dataset as ds  # type: ignore
from ruamel.yaml import YAML

from bystro.ancestry.ancestry_types import AncestryResults, AncestryJobCompleteMessage, AncestryJobData
from bystro.ancestry.inference import infer_ancestry
from bystro.beanstalkd.messages import (
    SubmittedJobMessage,
    ProgressPublisher,
    QueueConf,
    get_progress_reporter,
)
from bystro.beanstalkd.worker import listen

from bystro.ancestry.model import get_models

logging.basicConfig(
    filename="ancestry_listener.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

ANCESTRY_TUBE = "ancestry"


def _load_queue_conf(queue_conf_path: str) -> QueueConf:
    with Path(queue_conf_path).open(encoding="utf-8") as queue_config_file:
        raw_queue_conf = YAML(typ="safe").load(queue_config_file)
    beanstalk_conf = raw_queue_conf["beanstalkd"]
    return QueueConf(addresses=beanstalk_conf["addresses"], tubes=beanstalk_conf["tubes"])


def handler_fn(publisher: ProgressPublisher, job_data: AncestryJobData) -> AncestryResults:
    """Do ancestry job, wrapping infer_ancestry for beanstalk."""
    # Separating _handler_fn from infer_ancestry in order to separate ML from infra concerns,
    # and especially to keep infer_ancestry eager.

    # not doing anything with this reporter at the moment, we're
    # simply threading it through for later.
    _reporter = get_progress_reporter(publisher)

    dataset = ds.dataset(job_data.dosage_matrix_path, format="arrow")

    ancestry_models = get_models(job_data.assembly)

    return infer_ancestry(ancestry_models, dataset)


def submit_msg_fn(ancestry_job_data: AncestryJobData) -> SubmittedJobMessage:
    """Acknowledge receipt of AncestryJobData."""
    logger.debug("entering submit_msg_fn: %s", ancestry_job_data)
    return SubmittedJobMessage(ancestry_job_data.submission_id)


def completed_msg_fn(
    ancestry_job_data: AncestryJobData, results: AncestryResults
) -> AncestryJobCompleteMessage:
    """Send job complete message."""
    logger.debug("entering completed_msg_fn: %s", ancestry_job_data)

    json_data = msgspec.json.encode(results)

    out_path = str(Path(ancestry_job_data.out_dir) / "ancestry_results.json")

    with open(out_path, "wb") as f:
        f.write(json_data)

    return AncestryJobCompleteMessage(
        submission_id=ancestry_job_data.submission_id, result_path=out_path
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

    logger.info(
        "Ancestry worker is listening on addresses: %s, tube: %s...", queue_conf.addresses, ANCESTRY_TUBE
    )

    listen(
        job_data_type=AncestryJobData,
        handler_fn=handler_fn,
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        queue_conf=queue_conf,
        tube=ANCESTRY_TUBE,
    )
