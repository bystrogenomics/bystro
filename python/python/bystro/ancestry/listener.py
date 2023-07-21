"""Provide a worker for the ancestry model."""
import argparse
import logging
from pathlib import Path

import botocore
from ruamel.yaml import YAML
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from skops.io import load as skops_load

from bystro.ancestry.ancestry_types import (
    AncestryResponse,
    AncestrySubmission,
)
from bystro.ancestry.dummy_ancestry_response import make_dummy_ancestry_response
from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage
from bystro.beanstalkd.worker import ProgressPublisher, QueueConf, get_progress_reporter, listen

logging.basicConfig(filename="ancestry_listener.log", level=logging.DEBUG)
logger = logging.getLogger()

ANCESTRY_TUBE = "ancestry"


def _get_models_from_s3(
    s3_client: botocore.client.BaseClient,
) -> tuple[PCA, RandomForestClassifier]:
    ANCESTRY_BUCKET = "bystro-ancestry"
    PCA_FILE = "pca.skop"
    RFC_FILE = "rfc.skop"

    s3_client.download_file(ANCESTRY_BUCKET, PCA_FILE, PCA_FILE)
    s3_client.download_file(ANCESTRY_BUCKET, RFC_FILE, RFC_FILE)

    pca = skops_load(PCA_FILE)
    rfc = skops_load(RFC_FILE)
    return pca, rfc


class AncestryJobData(BaseMessage, frozen=True):
    """Wrap an AncestrySubmission in a BaseMessage for beanstalk."""

    ancestry_submission: AncestrySubmission


class AncestryJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    """Wrap an AncestryResponse in a CompletedJobMessage for beanstalk."""

    results: AncestryResponse


def _load_queue_conf(queue_conf_path: str) -> QueueConf:
    with Path(queue_conf_path).open(encoding="utf-8") as queue_config_file:
        raw_queue_conf = YAML(typ="safe").load(queue_config_file)
    beanstalk_conf = raw_queue_conf["beanstalkd"]
    return QueueConf(addresses=beanstalk_conf["addresses"], tubes=beanstalk_conf["tubes"])


# TODO: implement with ray
def _infer_ancestry(
    ancestry_submission: AncestrySubmission, _publisher: ProgressPublisher
) -> AncestryResponse:
    """Run an ancestry job."""
    # TODO: main ancestry model logic goes here.  Just stubbing out for now.
    logger.debug("Inferring ancestry for: %s", ancestry_submission)

    # not doing anything with this reporter at the moment, we're
    # simply threading it through for later.
    _reporter = get_progress_reporter(_publisher)
    vcf_path = ancestry_submission.vcf_path

    return make_dummy_ancestry_response(vcf_path)


def handler_fn(publisher: ProgressPublisher, ancestry_job_data: AncestryJobData) -> AncestryResponse:
    """Do ancestry job, wraping _infer_ancestry for beanstalk."""
    # Separating _handler_fn from _infer_ancestry in order to separate ML from infra concerns,
    # and especially to keep _infer_ancestry eager.
    logger.debug("entering handler_fn with: %s", ancestry_job_data)
    return _infer_ancestry(ancestry_job_data.ancestry_submission, publisher)


def submit_msg_fn(ancestry_job_data: AncestryJobData) -> SubmittedJobMessage:
    """Acknowledge receipt of AncestryJobData."""
    logger.debug("entering submit_msg_fn: %s", ancestry_job_data)
    return SubmittedJobMessage(ancestry_job_data.submissionID)


def completed_msg_fn(
    ancestry_job_data: AncestryJobData, ancestry_response: AncestryResponse
) -> AncestryJobCompleteMessage:
    """Send job complete message."""
    logger.debug("entering completed_msg_fn: %s", ancestry_job_data)
    ancestry_submission = ancestry_job_data.ancestry_submission
    if ancestry_submission.vcf_path != ancestry_response.vcf_path:
        err_msg = (
            f"Ancestry submission filename {ancestry_submission.vcf_path} "
            "doesn't match response filename {ancestry_response.vcf_path}: this is a bug."
        )
        raise ValueError(err_msg)
    logger.debug("completed ancestry inference for: %s", ancestry_response)
    return AncestryJobCompleteMessage(
        submissionID=ancestry_job_data.submissionID, results=ancestry_response
    )


def main(queue_conf: QueueConf, ancestry_tube: str) -> None:
    """Run ancestry listener."""
    logger.debug("Entering main")
    listen(
        AncestryJobData,
        handler_fn,
        submit_msg_fn,
        completed_msg_fn,
        queue_conf,
        ancestry_tube,
    )


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

    main(queue_conf, ancestry_tube=ANCESTRY_TUBE)
