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
map_cache = {}
sumstats_cache = {}

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


def get_sumstats_file(trait: str, assembly: str, pmid: str) -> Path:
    """
    Get the path to the sumstats file for a given trait/disease/outcome, assembly, and PubMed ID.

    Args:
        trait (str): The trait to use.
        assembly (str): The assembly to use.
        pmid (str): The PubMed ID of the study.

    Returns:

    """
    sumstats_cache_key = f"{trait}_{assembly}_{pmid}"

    if sumstats_cache_key in sumstats_cache:
        logger.debug("Sumstats for %s found in cache.", sumstats_cache_key)
    else:
        sumstats_file = sumstats_template % (trait, assembly, pmid)
        local_sumstats_file = PRS_MODEL_DIR / assembly / PRS_MODEL_SUMSTATS_SUBDIR / sumstats_file
        local_sumstats_dir = local_sumstats_file.parent
        local_sumstats_dir.mkdir(parents=True, exist_ok=True)

        remote_sumstats_file = f"{assembly}/{sumstats_file}"

        if not local_file_exists(str(local_sumstats_file)):
            with Timer() as timer:
                try:
                    download_file(
                        bucket=PRS_BUCKET, key=remote_sumstats_file, filename=str(local_sumstats_file)
                    )
                except requests.HTTPError as e:
                    raise ValueError(f"{sumstats_file} not found in bucket {PRS_BUCKET}.") from e

            logger.debug("Downloaded sumstats file in %f seconds", timer.elapsed_time)

        sumstats_cache[sumstats_cache_key] = local_sumstats_file

    return sumstats_cache[sumstats_cache_key]


def get_map_file(assembly: str, population: str) -> Path:
    """
    Get the path to the genetic map file for a given assembly and population.

    Args:
        assembly (str): The assembly to use.
        population (str): The population to use.

    Returns:

    """
    map_cache_key = f"{assembly}_{population}"

    if map_cache_key in map_cache:
        logger.debug("Genetic map for %s found in cache.", map_cache_key)
    else:
        map_file = map_template % (assembly, population)
        local_map_file = PRS_MODEL_DIR / assembly / PRS_MODEL_MAP_SUBDIR / map_file
        local_map_dir = local_map_file.parent
        local_map_dir.mkdir(parents=True, exist_ok=True)

        remote_map_file = f"{assembly}/{map_file}"

        if not local_file_exists(str(local_map_file)):
            with Timer() as timer:
                try:
                    download_file(bucket=PRS_BUCKET, key=remote_map_file, filename=str(local_map_file))
                except requests.HTTPError as e:
                    raise ValueError(f"{map_file} not found in bucket {PRS_BUCKET}.") from e

            logger.debug("Downloaded genetic map file in %f seconds", timer.elapsed_time)

        map_cache[map_cache_key] = local_map_file

    return map_cache[map_cache_key]


def local_file_exists(file_path: str) -> bool:
    """
    Check whether a file exists in the local filesystem.

    Args:
        file_path (str): The local path

    Returns:
        bool: True if the file exists, False otherwise.
    """

    if not os.path.exists(file_path):
        return False

    return True
