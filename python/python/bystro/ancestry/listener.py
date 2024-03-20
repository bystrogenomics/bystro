"""Provide a worker for the ancestry model."""

import argparse
import logging
from pathlib import Path

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
import msgspec
import pandas as pd
import pyarrow.dataset as ds  # type: ignore
from ruamel.yaml import YAML
from skops.io import load as skops_load  # type: ignore

from bystro.ancestry.ancestry_types import AncestryResults
from bystro.ancestry.inference import AncestryModel, AncestryModels, infer_ancestry
from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage
from bystro.beanstalkd.worker import ProgressPublisher, QueueConf, get_progress_reporter, listen

from bystro.utils.timer import Timer

logging.basicConfig(
    filename="ancestry_listener.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

ANCESTRY_TUBE = "ancestry"
ANCESTRY_BUCKET = "bystro-ancestry"
GNOMAD_PCA_FILE = "gnomadset_pca.csv"
GNOMAD_RFC_FILE = "gnomadset_rfc.skop"
ARRAY_PCA_FILE = "arrayset_pca.csv"
ARRAY_RFC_FILE = "arrayset_rfc.skop"

models_cache: dict[str, AncestryModels] = {}


def _get_one_model_from_s3(pca_local_key, rfc_local_key, pca_file_key, rfc_file_key) -> AncestryModel:
    s3_client = boto3.client("s3")

    logger.info("Downloading PCA file %s", pca_file_key)

    with Timer() as timer:
        try:
            s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=pca_file_key, Filename=pca_local_key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise ValueError(f"{pca_file_key} not found. This assembly is not supported.")
            raise  # Re-raise the exception if it's not a "NoSuchKey" error

        try:
            s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=rfc_file_key, Filename=rfc_local_key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise ValueError(
                    f"{rfc_file_key} ancestry model not found. This assembly is not supported."
                )
            raise

    logger.debug("Downloaded PCA file and RFC file in %f seconds", timer.elapsed_time)

    with Timer() as timer:
        logger.info("Loading PCA file %s", pca_local_key)
        pca_loadings_df = pd.read_csv(pca_local_key, index_col=0)

        logger.info("Loading RFC file %s", rfc_local_key)
        rfc = skops_load(rfc_local_key)

    logger.debug("Loaded PCA and RFC files in %f seconds", timer.elapsed_time)

    logger.info("Loaded ancestry models from S3")

    return AncestryModel(pca_loadings_df, rfc)


def _get_models_from_s3(assembly: str) -> AncestryModels:
    if assembly in models_cache:
        logger.info("Model for assembly %s found in cache.", assembly)
        return models_cache[assembly]

    pca_local_key_gnomad = f"{assembly}_{GNOMAD_PCA_FILE}"
    rfc_local_key_gnomad = f"{assembly}_{GNOMAD_RFC_FILE}"

    pca_file_key_gnomad = f"{assembly}/{pca_local_key_gnomad}"
    rfc_file_key_gnomad = f"{assembly}/{rfc_local_key_gnomad}"

    pca_local_key_array = f"{assembly}_{ARRAY_PCA_FILE}"
    rfc_local_key_array = f"{assembly}_{ARRAY_RFC_FILE}"

    pca_file_key_array = f"{assembly}/{pca_local_key_array}"
    rfc_file_key_array = f"{assembly}/{rfc_local_key_array}"

    gnomad_model = _get_one_model_from_s3(
        pca_local_key_gnomad, rfc_local_key_gnomad, pca_file_key_gnomad, rfc_file_key_gnomad
    )
    array_model = _get_one_model_from_s3(
        pca_local_key_array, rfc_local_key_array, pca_file_key_array, rfc_file_key_array
    )

    models = AncestryModels(gnomad_model, array_model)

    # Update the cache with the new model
    if len(models_cache) >= 1:
        # Remove the oldest loaded model to maintain cache size
        oldest_assembly = next(iter(models_cache))
        del models_cache[oldest_assembly]
    models_cache[assembly] = models

    return models


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
    assembly: str
        The genome assembly used for the dosage matrix.
    """

    dosage_matrix_path: str
    out_dir: str
    assembly: str


class AncestryJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True, rename="camel"):
    """The returned JSON message expected by the API server"""

    result_path: str


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

    ancestry_models = _get_models_from_s3(job_data.assembly)

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
