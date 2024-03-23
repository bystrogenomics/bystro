"""Provide a worker for the ancestry model."""

import logging
from pathlib import Path

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
import pandas as pd
from skops.io import load as skops_load  # type: ignore

from bystro.ancestry.inference import AncestryModel, AncestryModels

from bystro.utils.timer import Timer

logging.basicConfig(
    filename="ancestry_model.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

ANCESTRY_BUCKET = "bystro-ancestry"
GNOMAD_PCA_FILE = "gnomadset_pca.csv"
GNOMAD_RFC_FILE = "gnomadset_rfc.skop"
ARRAY_PCA_FILE = "arrayset_pca.csv"
ARRAY_RFC_FILE = "arrayset_rfc.skop"

models_cache: dict[str, AncestryModels] = {}


def get_one_model_from_s3(pca_local_path, rfc_local_path, pca_file_key, rfc_file_key) -> AncestryModel:
    """_summary_

    Args:
        pca_local_path (_type_): The local path to save the PCA file.
        rfc_local_path (_type_): The local path to save the RFC file.
        pca_file_key (_type_): The remove path to the PCA file.
        rfc_file_key (_type_): The remove path to the RFC file.

    Raises:
        ValueError: If the PCA or RFC file is not found.

    Returns:
        AncestryModel: The loaded ancestry model.
    """
    s3_client = boto3.client("s3")

    logger.info("Downloading PCA file %s", pca_file_key)

    with Timer() as timer:
        try:
            s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=pca_file_key, Filename=pca_local_path)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise ValueError(f"{pca_file_key} not found. This assembly is not supported.") from e
            raise  # Re-raise the exception if it's not a "NoSuchKey" error

        try:
            s3_client.download_file(Bucket=ANCESTRY_BUCKET, Key=rfc_file_key, Filename=rfc_local_path)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise ValueError(
                    f"{rfc_file_key} ancestry model not found. This assembly is not supported."
                ) from e
            raise

    logger.debug("Downloaded PCA file and RFC file in %f seconds", timer.elapsed_time)

    return get_one_model_from_file_system(pca_local_path, rfc_local_path)


def get_models_from_s3(assembly: str) -> AncestryModels:
    """
    Load the ancestry models for the given assembly from S3.

    Args:
        assembly (str): The genome assembly to load the models for.
            Example: "hg38" or "hg19"

    Returns:
        AncestryModels: The loaded models for the given assembly.
    """
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

    gnomad_model = get_one_model_from_s3(
        pca_local_key_gnomad, rfc_local_key_gnomad, pca_file_key_gnomad, rfc_file_key_gnomad
    )
    array_model = get_one_model_from_s3(
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


def get_one_model_from_file_system(pca_path: str, rfc_path: str) -> AncestryModel:
    """
    Load an ancestry model from the local file system.

    Args:
        pca_path (str): The path to the PCA file.
        rfc_path (str): The path to the RFC file.

    Returns:
        AncestryModel: The loaded ancestry model.
    """
    with Timer() as timer:
        logger.info("Loading PCA file %s", pca_path)
        pca_loadings_df = pd.read_csv(pca_path, index_col=0)

        logger.info("Loading RFC file %s", rfc_path)
        rfc = skops_load(rfc_path)

    logger.debug("Loaded PCA and RFC files in %f seconds", timer.elapsed_time)

    return AncestryModel(pca_loadings_df, rfc)


def get_models_from_file_system(model_dir: str, assembly: str) -> AncestryModels:
    """
    Load the ancestry models for the given assembly from the local file system.

    Args:
        model_dir (str): The local directory where the models are stored.
            We expect the models to be stored in the following format:
            ```
            model_dir/
                assembly/
                    gnomadset_pca.csv
                    gnomadset_rfc.skop
                    arrayset_pca.csv
                    arrayset_rfc.skop
            ```
        assembly (str): The genome assembly to load the models for.
            Example: "hg38" or "hg19"

    Returns:
        AncestryModels: The loaded models for the given assembly.
    """
    if assembly in models_cache:
        logger.info("Model for assembly %s found in cache.", assembly)
        return models_cache[assembly]

    pca_local_key_gnomad = str(Path(model_dir) / assembly / f"{assembly}_{GNOMAD_PCA_FILE}")
    rfc_local_key_gnomad = str(Path(model_dir) / assembly / f"{assembly}_{GNOMAD_RFC_FILE}")

    pca_local_key_array = str(Path(model_dir) / assembly / f"{assembly}_{ARRAY_PCA_FILE}")
    rfc_local_key_array = str(Path(model_dir) / assembly / f"{assembly}_{ARRAY_RFC_FILE}")

    gnomad_model = get_one_model_from_file_system(pca_local_key_gnomad, rfc_local_key_gnomad)
    array_model = get_one_model_from_file_system(pca_local_key_array, rfc_local_key_array)

    models = AncestryModels(gnomad_model, array_model)

    # Update the cache with the new model
    if len(models_cache) >= 1:
        # Remove the oldest loaded model to maintain cache size
        oldest_assembly = next(iter(models_cache))
        del models_cache[oldest_assembly]
    models_cache[assembly] = models

    return models
