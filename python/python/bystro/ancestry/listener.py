"""Provide a worker for the ancestry model."""
import argparse
import logging
import os
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

import boto3
import botocore
import pandas as pd
from ruamel.yaml import YAML
from sklearn.ensemble import RandomForestClassifier
from skops.io import load as skops_load

from bystro.ancestry.ancestry_types import (
    AncestryResponse,
    AncestryResult,
    AncestrySubmission,
    PopulationVector,
    ProbabilityInterval,
    SuperpopVector,
)
from bystro.ancestry.train import POPS, parse_vcf, superpop_probs_from_pop_probs
from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage
from bystro.beanstalkd.worker import ProgressPublisher, QueueConf, get_progress_reporter, listen

logging.basicConfig(filename="ancestry_listener.log", level=logging.DEBUG)
logger = logging.getLogger()

ANCESTRY_TUBE = "ancestry"
ANCESTRY_BUCKET = "bystro-ancestry"
PCA_FILE = "pca.skop"
RFC_FILE = "rfc.skop"


def _check_vcf_dir_access(vcf_dir: str) -> None:
    try:
        os.listdir(vcf_dir)
    except FileNotFoundError as err:
        err_msg = (
            f"Couldn't access VCF dir {vcf_dir}, "
            "will not be able to read VCFs in order to report ancestry results. "
            "Check whether EFS is mounted correctly?"
        )
        raise FileNotFoundError(err_msg) from err


@dataclass(frozen=True)
class AncestryModel:
    """Bundle together PCA and RFC models for bookkeeping purposes."""

    pca_loadings_df: pd.DataFrame
    rfc: RandomForestClassifier

    def __post_init__(self) -> "AncestryModel":
        """Ensure that PCA and RFC features line up correctly."""
        pca_cols = self.pca_loadings_df.columns
        rfc_features = self.rfc.feature_names_in_
        if not (len(pca_cols) == len(rfc_features) and (pca_cols == rfc_features).all()):
            err_msg = (
                f"PC loadings columns:{self.pca_loadings_df.columns} must equal "
                f"rfc.feature_names_in: {self.rfc.feature_names_in_}"
            )
            raise ValueError(err_msg)
        return self

    def predict_proba(self, genotypes: pd.DataFrame) -> pd.DataFrame:
        """Predict population probabilities from dosage matrix."""
        Xpc = genotypes @ self.pca_loadings_df
        probs = self.rfc.predict_proba(Xpc)
        return pd.DataFrame(probs, index=genotypes.index, columns=POPS)


def _get_model_from_s3(
    s3_client: botocore.client.BaseClient,
) -> AncestryModel:
    s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=PCA_FILE, Filename=PCA_FILE)
    s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=RFC_FILE, Filename=RFC_FILE)

    pca_loadings_df = pd.read_csv(PCA_FILE, index_col=0)
    rfc = skops_load(RFC_FILE)
    logger.info("Loaded ancestry models from S3")
    return AncestryModel(pca_loadings_df, rfc)


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


def _fill_missing_data(genotypes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missingnesses = genotypes.isna().mean(axis="columns")
    # todo: much better imputation strategy to come, but we're stubbing out for now.
    imputed_genotypes = genotypes.fillna(genotypes.mean())
    return imputed_genotypes, missingnesses


def _load_vcf(full_vcf_path: Path, variants: Collection[str]) -> pd.DataFrame:
    """Load vcf, return dosages as df where index is sample_ids, columns are variants."""
    # Currently the implementation is trivial, but we're stubbing this
    # out now in order to encapsulate future volatility arising from EFS handling, &c.
    logger.info("loading vcf from %s", full_vcf_path)
    return parse_vcf(full_vcf_path, variants)


def _make_trivial_probability_interval(x: float) -> ProbabilityInterval:
    """Promote a value to a trivial ProbabilityInterval with equal lower, upper bounds."""
    return ProbabilityInterval(x, x)


def _package_ancestry_response_from_pop_probs(
    vcf_path: str, pop_probs_df: pd.DataFrame, missingnesses: pd.Series
) -> AncestryResponse:
    superpop_probs_df = superpop_probs_from_pop_probs(pop_probs_df)
    ancestry_results = []

    for (sample_id, sample_pop_probs), (_sample_id2, sample_superpop_probs) in zip(
        pop_probs_df.iterrows(), superpop_probs_df.iterrows(), strict=True
    ):
        if not isinstance(sample_id, str):
            raise TypeError
        pop_vector = PopulationVector(
            **{
                pop: _make_trivial_probability_interval(value)
                for (pop, value) in dict(sample_pop_probs).items()
            }
        )
        superpop_vector = SuperpopVector(
            **{
                superpop: _make_trivial_probability_interval(value)
                for (superpop, value) in dict(sample_superpop_probs).items()
            }
        )
        ancestry_results.append(
            AncestryResult(
                sample_id=sample_id,
                populations=pop_vector,
                superpops=superpop_vector,
                missingness=missingnesses[sample_id],
            )
        )
    return AncestryResponse(vcf_path=vcf_path, results=ancestry_results)


# TODO: implement with ray
def _infer_ancestry(
    ancestry_model: AncestryModel, genotypes: pd.DataFrame, vcf_path: str
) -> AncestryResponse:
    """Run an ancestry job."""
    # TODO: main ancestry model logic goes here.  Just stubbing out for now.

    imputed_genotypes, missingnesses = _fill_missing_data(genotypes)
    pop_probs_df = ancestry_model.predict_proba(imputed_genotypes)
    return _package_ancestry_response_from_pop_probs(vcf_path, pop_probs_df, missingnesses)


def handler_fn_factory(
    ancestry_model: AncestryModel, vcf_dir: Path
) -> Callable[[ProgressPublisher, AncestryJobData], AncestryResponse]:
    """Partial handler_fn to accept an ancestry_model."""

    def handler_fn(publisher: ProgressPublisher, ancestry_job_data: AncestryJobData) -> AncestryResponse:
        """Do ancestry job, wrapping _infer_ancestry for beanstalk."""
        # Separating _handler_fn from _infer_ancestry in order to separate ML from infra concerns,
        # and especially to keep _infer_ancestry eager.

        # not doing anything with this reporter at the moment, we're
        # simply threading it through for later.
        _reporter = get_progress_reporter(publisher)
        logger.debug("entering handler_fn with: %s", ancestry_job_data)
        vcf_path = ancestry_job_data.ancestry_submission.vcf_path
        full_vcf_path = vcf_dir / vcf_path
        genotypes = _load_vcf(full_vcf_path, variants=ancestry_model.pca_loadings_df.index)
        return _infer_ancestry(ancestry_model, genotypes, vcf_path)

    return handler_fn


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


def main(ancestry_model: AncestryModel, vcf_dir: Path, queue_conf: QueueConf) -> None:
    """Run ancestry listener."""
    logger.debug("Entering main")
    handler_fn_with_models = handler_fn_factory(ancestry_model, vcf_dir)
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
    parser.add_argument(
        "--vcf-dir",
        type=Path,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
        required=True,
    )

    args = parser.parse_args()

    _check_vcf_dir_access(args.vcf_dir)
    s3_client = boto3.client("s3")
    ancestry_model = _get_model_from_s3(s3_client)
    queue_conf = _load_queue_conf(args.queue_conf)

    main(ancestry_model, args.vcf_dir, queue_conf)
