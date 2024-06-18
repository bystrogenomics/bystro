import logging
import os
from pathlib import Path
import requests

from msgspec import Struct

from bystro.utils.timer import Timer

logging.basicConfig(
    filename="prs_model.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

PRS_BUCKET = os.getenv("PRS_BUCKET", "bystro-prs-public")
PRS_MODEL_DIR = Path(os.getenv("PRS_MODEL_DIR", str(Path(__file__).parent / "data")))
PRS_MODEL_MAP_SUBDIR = "map"
PRS_MODEL_SUMSTATS_SUBDIR = "sumstats"

class PrsModel(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    map_path: str
    score_path: str

models_cache: dict[str, PrsModel] = {}

map_template = "%s_genetic_map_%s.feather"
sumstats_template = "%s_sumstats_%s_%s_compressed.feather"

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


def get_one_model(assembly: str, population: str, disease: str, pmid: str) -> PrsModel:
    """
    Load an ancestry model from S3.

    Args:
        assembly (str): The assembly to use.
        population (str): The population to use.
        disease (str): The disease to use.
        pmid (str): The PubMed ID of the study.

    Raises:
        ValueError: If the map or sumstats file is not found.

    Returns:
        PrsModel: The loaded PRS model.
    """
    logger.info("Downloading PRS files for %s %s %s %s", assembly, population, disease, pmid)

    # Check if in cache
    key = f"{assembly}_{population}_{disease}_{pmid}"
    if key in models_cache:
        logger.debug("PRS model found in cache.")
        return models_cache[key]

    map_file = map_template % (assembly, population)
    sumstats_file = sumstats_template % (disease, assembly, pmid)

    local_map_file = PRS_MODEL_DIR / assembly / PRS_MODEL_MAP_SUBDIR / map_file
    local_map_dir = local_map_file.parent
    local_map_dir.mkdir(parents=True, exist_ok=True)

    remote_map_file = f"{assembly}/{map_file}"

    local_sumstats_file = PRS_MODEL_DIR / assembly / PRS_MODEL_SUMSTATS_SUBDIR / sumstats_file
    local_sumstats_dir = local_sumstats_file.parent
    local_sumstats_dir.mkdir(parents=True, exist_ok=True)

    remote_sumstats_file = f"{assembly}/{sumstats_file}"

    with Timer() as timer:
        try:
            prs_model = get_one_model_from_file_system(str(local_map_file), str(local_sumstats_file))
            logger.debug("PRS model found on file system.")
        except ValueError:
            logger.debug("PRS model not found on file system. Downloading from S3.")

            try:
                download_file(bucket=PRS_BUCKET, key=remote_map_file, filename=str(local_map_file))
            except requests.HTTPError as e:
                raise ValueError(f"{map_file} not found in bucket {PRS_BUCKET}.") from e

            try:
                download_file(bucket=PRS_BUCKET, key=remote_sumstats_file, filename=str(local_sumstats_file))
            except requests.HTTPError as e:
                raise ValueError(f"{sumstats_file} not found in bucket {PRS_BUCKET}.") from e

            prs_model = get_one_model_from_file_system(str(local_map_file), str(local_sumstats_file))

    logger.debug("Downloaded PCA file and RFC file in %f seconds", timer.elapsed_time)

    models_cache[key] = prs_model

    return prs_model


def get_one_model_from_file_system(local_map_file: str, local_sumstats_file: str) -> PrsModel:
    """
    Load a PRS model from the local file system.

    Args:
        local_map_file (str): The local path to the genetic map file.
        local_sumstats_file (str): The local path to the sumstats file.

    Returns:
        PrsModel: The loaded PRS model.
    """

    if not os.path.exists(local_map_file):
        raise ValueError(f"Genetic map file {local_map_file} not found.")

    if not os.path.exists(local_sumstats_file):
        raise ValueError(f"Sumstats file {local_sumstats_file} not found.")

    return PrsModel(map_path=local_map_file, score_path=local_sumstats_file)
