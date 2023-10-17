"""Provide a worker for PRS calculation."""

import argparse
import logging
from pathlib import Path

from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage
from bystro.beanstalkd.worker import ProgressPublisher, QueueConf, get_progress_reporter, listen
from bystro.utils.timer import Timer

# TODO: Define PRS calculation and preprocess covariate, association files
from bystro.prs.prs_types import PRSResponse, PRSSubmission
from bystro.prs.calculate_prs import (
    _load_dosage_matrix,
    _load_association_scores,
    _load_covariates,
    calculate_prs_scores,
)

# TODO: Define how we load vcf-like genotype matrix from annotation,
# then generalize this process for other analyses
from bystro.prs.utils import load_vcf_from_annotation

# TODO: Implement function to preprocess genotype data from annotation
# into format suitable for PRS calculation
from bystro.prs.utils import preprocess_vcf_for_prs

from ruamel.yaml import YAML

logging.basicConfig(
    filename="prs_worker.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

PRS_TUBE = "prs"


class PRSJobData(BaseMessage, frozen=True):
    """Wrap a PRSSubmission in a BaseMessage for beanstalkd."""

    prs_submission: PRSSubmission


class PRSJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    """Wrap a PRSResponse in a CompletedJobMessage for beanstalkd."""

    results: PRSResponse


def _load_queue_conf(queue_conf_path: str) -> QueueConf:
    with Path(queue_conf_path).open(encoding="utf-8") as queue_config_file:
        raw_queue_conf = YAML(typ="safe").load(queue_config_file)
    beanstalk_conf = raw_queue_conf["beanstalkd"]
    return QueueConf(addresses=beanstalk_conf["addresses"], tubes=beanstalk_conf["tubes"])


def handler_fn(publisher: ProgressPublisher, prs_job_data: PRSJobData) -> PRSResponse:
    _reporter = get_progress_reporter(publisher)
    logger.debug("entering handler_fn with: %s", prs_job_data)
    dosage_matrix_path = prs_job_data.prs_submission.dosage_matrix_path
    association_scores_path = prs_job_data.prs_submission.association_score_filepath
    covariate_path = prs_job_data.prs_submission.covariate_filepath
    logger.debug("loading genotype matrix %s", dosage_matrix_path)
    with Timer() as timer:
        # TODO: Load 3 files in
        genotypes = _load_dosage_matrix(dosage_matrix_path)
        scores = _load_association_scores(association_scores_path)
        covariates = _load_covariates(covariate_path)
        parameters = "default"
    logger.debug(
        "finished loading genotype matrix %s in %f seconds", dosage_matrix_path, timer.elapsed_time
    )
    prs_calculation = calculate_prs_scores(scores, covariates, genotypes, parameters)
    return prs_calculation


def submit_msg_fn(prs_job_data: PRSJobData) -> SubmittedJobMessage:
    """Acknowledge receipt of PRS Job Data."""
    logger.debug("entering submit_msg_fn: %s", prs_job_data)
    return SubmittedJobMessage(prs_job_data.submissionID)


def completed_msg_fn(prs_job_data: PRSJobData, prs_response: PRSResponse) -> PRSJobCompleteMessage:
    """Send job completed message."""
    logger.debug("entering completed_msg_fn: %s", prs_job_data)
    prs_submission = prs_job_data.prs_submission
    if prs_submission.dosage_matrix_path != prs_response.dosage_matrix_path:
        err_msg = (
            f"PRS submission filename {prs_submission.dosage_matrix_path} "
            f"doesn't match response filename {prs_response.dosage_matrix_path}: this is a bug."
        )
        raise ValueError(err_msg)
    logger.debug("completed prs calculation for: %s", prs_response)
    return PRSJobCompleteMessage(submissionID=prs_job_data.submissionID, results=prs_response)


def main(beanstalk_host: str, beanstalk_port: int, covariate_file: Path, association_file: Path) -> None:
    """Run PRS listener."""
    prs_handler_fn = handler_fn
    logger.info("PRS worker is listening on addresses: %s, tube: %s...", queue_conf.addresses, PRS_TUBE)
    listen(
        PRSJobData,
        prs_handler_fn,
        submit_msg_fn,
        completed_msg_fn,
        queue_conf,
        PRS_TUBE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRS Beanstalkd Worker")
    parser.add_argument(
        "--queue_conf",
        type=Path,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
        required=True,
    )
    parser.add_argument(
        "--covariate_file",
        type=Path,
        required=True,
        help="Path to the covariate file for PRS",
    )
    parser.add_argument(
        "--association_file",
        type=Path,
        required=True,
        help="Path to the association file for PRS",
    )
    args = parser.parse_args()

    queue_conf = _load_queue_conf(args.queue_conf)

    main(args.beanstalk_host, args.beanstalk_port, args.covariate_file, args.association_file)
