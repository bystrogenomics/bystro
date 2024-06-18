"""Provide a worker for the ancestry model."""

import logging
import os
from pathlib import Path
import requests

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

pd.options.future.infer_string = True  # type: ignore

ANCESTRY_BUCKET = os.getenv("ANCESTRY_BUCKET", "bystro-ancestry-public")
ANCESTRY_MODEL_DIR = os.getenv("ANCESTRY_MODEL_DIR", str(Path(__file__).parent / "data"))
GNOMAD_PCA_FILE = "gnomadset_pca.csv"
GNOMAD_RFC_FILE = "gnomadset_rfc.skop"
ARRAY_PCA_FILE = "arrayset_pca.csv"
ARRAY_RFC_FILE = "arrayset_rfc.skop"

models_cache: dict[str, AncestryModels] = {}


def download_file(bucket: str, key: str, filename: str):
    """
    Download a file from the given URL to the local path.

    Args:
        url (str): The URL to download the file from.
        local_path (str): The local path to save the file.
    """
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    logger.info("Downloading file from %s to %s", url, filename)
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    with open(filename, "wb") as f:
        f.write(response.content)


def get_one_model(
    pca_local_path: str, rfc_local_path: str, pca_file_key: str, rfc_file_key: str
) -> AncestryModel:
    """
    Load an ancestry model from S3.

    Args:
        pca_local_path (str): The local path to save the PCA file.
        rfc_local_path (str): The local path to save the RFC file.
        pca_file_key (str): The remove path to the PCA file.
        rfc_file_key (str): The remove path to the RFC file.

    Raises:
        ValueError: If the PCA or RFC file is not found.

    Returns:
        AncestryModel: The loaded ancestry model.
    """
    logger.info(
        "Downloading PCA file %s and RFC file %s to %s and %s",
        pca_file_key,
        rfc_file_key,
        pca_local_path,
        rfc_local_path,
    )

    with Timer() as timer:
        try:
            download_file(bucket=ANCESTRY_BUCKET, key=pca_file_key, filename=pca_local_path)
        except requests.HTTPError as e:
            raise ValueError(
                f"{pca_file_key} not found in bucket {ANCESTRY_BUCKET}. "
                "This assembly is not supported."
            ) from e

        try:
            download_file(bucket=ANCESTRY_BUCKET, key=rfc_file_key, filename=rfc_local_path)
        except requests.HTTPError as e:
            raise ValueError(
                f"{rfc_file_key} not found in bucket {ANCESTRY_BUCKET}. "
                "This assembly is not supported."
            ) from e

    logger.debug("Downloaded PCA file and RFC file in %f seconds", timer.elapsed_time)

    return get_one_model_from_file_system(pca_local_path, rfc_local_path)


def get_models(assembly: str) -> AncestryModels:
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

    paths = _get_local_paths(assembly)

    if (
        Path(paths["gnomad"]["pca_local_path"]).exists()
        and Path(paths["gnomad"]["rfc_local_path"]).exists()
        and Path(paths["array"]["pca_local_path"]).exists()
        and Path(paths["array"]["rfc_local_path"]).exists()
    ):
        logger.info("Loading models from disk.")
        gnomad_model = get_one_model_from_file_system(
            paths["gnomad"]["pca_local_path"], paths["gnomad"]["rfc_local_path"]
        )
        array_model = get_one_model_from_file_system(
            paths["array"]["pca_local_path"], paths["array"]["rfc_local_path"]
        )
        models = AncestryModels(gnomad_model, array_model)
    else:
        gnomad_model = get_one_model(
            paths["gnomad"]["pca_local_path"],
            paths["gnomad"]["rfc_local_path"],
            paths["gnomad"]["pca_remote_path"],
            paths["gnomad"]["rfc_remote_path"],
        )
        array_model = get_one_model(
            paths["array"]["pca_local_path"],
            paths["array"]["rfc_local_path"],
            paths["array"]["pca_remote_path"],
            paths["array"]["rfc_remote_path"],
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


def get_models_from_file_system(assembly: str) -> AncestryModels:
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

    paths = _get_local_paths(assembly)

    gnomad_model = get_one_model_from_file_system(
        paths["gnomad"]["pca_local_path"], paths["gnomad"]["rfc_local_path"]
    )
    array_model = get_one_model_from_file_system(
        paths["array"]["pca_local_path"], paths["array"]["rfc_local_path"]
    )

    models = AncestryModels(gnomad_model, array_model)

    # Update the cache with the new model
    if len(models_cache) >= 1:
        # Remove the oldest loaded model to maintain cache size
        oldest_assembly = next(iter(models_cache))
        del models_cache[oldest_assembly]
    models_cache[assembly] = models

    return models


def _get_local_paths(assembly: str) -> dict[str, dict[str, str]]:
    local_dir = Path(ANCESTRY_MODEL_DIR) / assembly
    local_dir.mkdir(exist_ok=True, parents=True)

    gnomad_pca_basename = f"{assembly}_{GNOMAD_PCA_FILE}"
    gnomad_rfc_basename = f"{assembly}_{GNOMAD_RFC_FILE}"
    array_pca_basename = f"{assembly}_{ARRAY_PCA_FILE}"
    array_rfc_basename = f"{assembly}_{ARRAY_RFC_FILE}"

    pca_local_path_gnomad = local_dir / gnomad_pca_basename
    rfc_local_path_gnomad = local_dir / gnomad_rfc_basename
    pca_local_path_array = local_dir / array_pca_basename
    rfc_local_path_array = local_dir / array_rfc_basename

    pca_remote_path_gnomad = f"{assembly}/{gnomad_pca_basename}"
    rfc_remote_path_gnomad = f"{assembly}/{gnomad_rfc_basename}"
    pca_remote_path_array = f"{assembly}/{array_pca_basename}"
    rfc_remote_path_array = f"{assembly}/{array_rfc_basename}"

    return {
        "gnomad": {
            "pca_local_path": str(pca_local_path_gnomad),
            "rfc_local_path": str(rfc_local_path_gnomad),
            "pca_remote_path": pca_remote_path_gnomad,
            "rfc_remote_path": rfc_remote_path_gnomad,
            "pca_basename": gnomad_pca_basename,
            "rfc_basename": gnomad_rfc_basename,
        },
        "array": {
            "pca_local_path": str(pca_local_path_array),
            "rfc_local_path": str(rfc_local_path_array),
            "pca_remote_path": pca_remote_path_array,
            "rfc_remote_path": rfc_remote_path_array,
            "pca_basename": array_pca_basename,
            "rfc_basename": array_rfc_basename,
        },
    }
