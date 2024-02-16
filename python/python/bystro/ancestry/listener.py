"""Provide a worker for the ancestry model."""

import argparse
from collections.abc import Callable
import logging
from pathlib import Path

import boto3  # type: ignore
import msgspec
import pandas as pd
import pyarrow.dataset as ds  # type: ignore
from ruamel.yaml import YAML
from skops.io import load as skops_load  # type: ignore

from bystro.ancestry.ancestry_types import AncestryResults
from bystro.ancestry.inference import AncestryModel, infer_ancestry
from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage
from bystro.beanstalkd.worker import ProgressPublisher, QueueConf, get_progress_reporter, listen

logging.basicConfig(
    filename="ancestry_listener.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

ANCESTRY_TUBE = "ancestry"
ANCESTRY_BUCKET = "bystro-ancestry"
PCA_FILE = "pca.csv"
RFC_FILE = "rfc.skop"


def _get_model_from_s3() -> AncestryModel:
    s3_client = boto3.client("s3")

    s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=PCA_FILE, Filename=PCA_FILE)
    s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=RFC_FILE, Filename=RFC_FILE)

    logger.info("Loading PCA file %s", PCA_FILE)
    pca_loadings_df = pd.read_csv(PCA_FILE, index_col=0)
    logger.info("Loading RFC file %s", RFC_FILE)
    rfc = skops_load(RFC_FILE)
    logger.info("Loaded ancestry models from S3")
    return AncestryModel(pca_loadings_df, rfc)


class AncestryJobData(BaseMessage, frozen=True, rename="camel"):
    """
    The expected JSON message for the Ancestry job.

    Parameters
    ----------
    submission_id: str
        The unique identifier for the job.
    dosage_matrix_path: str
        The path to the dosage matrix file.
    out_dir: str
        The directory to write the results to.
    """

    dosage_matrix_path: str
    out_dir: str


class AncestryJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True, rename="camel"):
    """The returned JSON message expected by the API server"""

    result_path: str


def _load_queue_conf(queue_conf_path: str) -> QueueConf:
    with Path(queue_conf_path).open(encoding="utf-8") as queue_config_file:
        raw_queue_conf = YAML(typ="safe").load(queue_config_file)
    beanstalk_conf = raw_queue_conf["beanstalkd"]
    return QueueConf(addresses=beanstalk_conf["addresses"], tubes=beanstalk_conf["tubes"])


def handler_fn_factory(
    ancestry_model: AncestryModel,
) -> Callable[[ProgressPublisher, AncestryJobData], AncestryResults]:
    """Return partialed handler_fn with ancestry_model loaded."""

    def handler_fn(publisher: ProgressPublisher, job_data: AncestryJobData) -> AncestryResults:
        """Do ancestry job, wrapping infer_ancestry for beanstalk."""
        # Separating _handler_fn from infer_ancestry in order to separate ML from infra concerns,
        # and especially to keep infer_ancestry eager.

        # not doing anything with this reporter at the moment, we're
        # simply threading it through for later.
        _reporter = get_progress_reporter(publisher)

        dataset = ds.dataset(job_data.dosage_matrix_path, format="arrow")

        return infer_ancestry(ancestry_model, dataset)

    return handler_fn


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


def main(ancestry_model: AncestryModel, queue_conf: QueueConf) -> None:
    """Run ancestry listener."""
    handler_fn_with_models = handler_fn_factory(ancestry_model)
    logger.info(
        "Ancestry worker is listening on addresses: %s, tube: %s...", queue_conf.addresses, ANCESTRY_TUBE
    )
    listen(
        AncestryJobData,
        handler_fn_with_models,
        submit_msg_fn,
        completed_msg_fn,
        queue_conf,
        ANCESTRY_TUBE,
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

    ancestry_model = _get_model_from_s3()
    queue_conf = _load_queue_conf(args.queue_conf)

    main(ancestry_model, queue_conf)
