"""Utility functions for accessing project-level configuration files."""
from enum import Enum
from pathlib import Path
from typing import Dict, Union

from ruamel.yaml import YAML

# A ConfigDict has string keys and values of type ConfigDictValue, or
# a list of ConfigDictValues.  A ConfigDictValue is a str, int, float,
# or (nested) ConfigDict.  ConfigDictValue can be extended with
# additional primitive types if necessary.


ConfigDict = Dict[str, Union["ConfigDictValue", list["ConfigDictValue"]]]
ConfigDictValue = Union[str, int, float, "ConfigDict"]


def _get_bystro_project_root() -> Path:
    """Return Path of bystro project root."""
    # find project root by walking up the tree until we get to top level bystro directory.
    # The bystro top level is assumed to be the unique directory containing a startup.yml file.
    path = Path().absolute()
    FILESYSTEM_ROOT = Path("/")
    found_startup_file = False
    while path != FILESYSTEM_ROOT:
        if any(path.glob("startup.yml")):
            found_startup_file = True
            break
        path = path.parent
    if not found_startup_file:
        msg = "Recursed to filesystem root without finding startup.yml file: this is a bug."
        raise FileNotFoundError(msg)
    return path


BYSTRO_PROJECT_ROOT = _get_bystro_project_root()
BYSTRO_CONFIG_DIR = BYSTRO_PROJECT_ROOT / "config"
OPENSEARCH_CONFIG_PATH = BYSTRO_CONFIG_DIR / "opensearch.yml"


def get_opensearch_config() -> ConfigDict:
    """Read opensearch config and return parsed YAML."""
    with OPENSEARCH_CONFIG_PATH.open() as search_config_file:
        config_dict: ConfigDict = YAML(typ="safe").load(search_config_file)
    return config_dict

class ReferenceGenome(Enum):
    """The collection of valid reference genomes."""

    ce11 = "ce11"
    dm6 = "dm6"
    hg19 = "hg19"
    hg19_ensembl = "hg19_ensembl"
    hg38 = "hg38"
    mm10 = "mm10"
    mm9 = "mm9"
    rheMac8 = "rheMac8"  # noqa: N815  (mixed case is necessary here)
    rn6 = "rn6"
    sacCer3 = "sacCer3"  # noqa: N815


def get_mapping_config(reference_genome: ReferenceGenome = ReferenceGenome.hg38) -> ConfigDict:
    """Load mapping config for given reference genome from YAML file."""
    base_name = f"{reference_genome.name}.mapping.yml"
    mapping_config_filepath = BYSTRO_CONFIG_DIR / base_name
    with Path.open(mapping_config_filepath, encoding="utf-8") as f:
        mapping_config = YAML(typ="safe").load(f)
    return mapping_config
