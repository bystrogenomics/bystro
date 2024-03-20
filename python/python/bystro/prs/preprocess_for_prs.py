"""Process dosage matrix, GWAS summary statistics, and LD maps for PRS C+T calculation."""

import glob
import logging
import re
import pandas as pd
from pyarrow import feather  # type: ignore

logger = logging.getLogger(__name__)

AD_SCORE_FILEPATH = "gwas_summary_stats/AD_sumstats_PMID35379992.feather"


def _load_preprocessed_dosage_matrix(dosage_matrix_path: str) -> pd.DataFrame:
    """Load in dosage matrix generated from bystro-vcf."""
    return feather.read_feather(dosage_matrix_path)


def _load_association_scores(AD_SCORE_FILEPATH: str) -> pd.DataFrame:
    """Load in GWAS summary statistics provided by user."""
    # For now, we are supporting one AD dataset PMID:35379992
    ad_scores = feather.read_feather(AD_SCORE_FILEPATH)
    columns_to_include = ["CHR", "POS", "OTHER_ALLELE", "EFFECT_ALLELE", "P", "SNPID", "BETA"]
    return ad_scores[columns_to_include]


def _load_genetic_maps_from_feather(map_directory_path: str) -> dict[str, pd.DataFrame]:
    """Load genetic maps from Feather files in the specified directory, using logging for messages.

    Args:
    ----
    map_directory_path: The path to the directory containing Feather files.

    Returns:
    -------
    A dictionary where keys are 'GeneticMap{chromosome_number}' and values are the corresponding
    DataFrames loaded from Feather files.
    """
    file_pattern = f"{map_directory_path}/*.feather"
    map_file_list = glob.glob(file_pattern)
    genetic_maps = {}
    for file in map_file_list:
        match = re.search(r"chromosome_(\d+)_genetic_map", file)
        if match:
            try:
                chrom_num = match.group(1)
                genetic_map = pd.read_feather(file)
                key = f"GeneticMap{chrom_num}"
                genetic_maps[key] = genetic_map
                logging.info("Successfully read %s from %s", key, file)
            except Exception as e:
                logging.exception("Failed to read from %s", file)
                raise RuntimeError(f"Failed to process {file} due to an error.") from e
        else:
            raise ValueError(
                "File format must match 'chromosome_1_genetic_map'. "
                f"Could not determine chromosome number from {file}."
            )

    return genetic_maps


def _preprocess_scores(AD_SCORE_FILEPATH: str) -> pd.DataFrame:
    """Process GWAS summary statistics to use effect scores for PRS."""
    # For now, we are supporting one AD dataset PMID:35379992

    ad_scores = _load_association_scores(AD_SCORE_FILEPATH)
    columns_to_include = ["CHR", "POS", "OTHER_ALLELE", "EFFECT_ALLELE", "P", "SNPID", "BETA"]
    preprocessed_scores = ad_scores[columns_to_include].copy()

    missing_columns = [col for col in columns_to_include if col not in ad_scores.columns]
    if missing_columns:
        raise ValueError(
            "The following required columns are missing from the GWAS summary statistics: "
            f"{', '.join(missing_columns)}"
        )
    preprocessed_scores["ID_effect_as_alt"] = (
        "chr"
        + preprocessed_scores["CHR"].astype(str)
        + ":"
        + preprocessed_scores["POS"].astype(str)
        + ":"
        + preprocessed_scores["OTHER_ALLELE"]
        + ":"
        + preprocessed_scores["EFFECT_ALLELE"]
    )

    preprocessed_scores["ID_effect_as_ref"] = (
        "chr"
        + preprocessed_scores["CHR"].astype(str)
        + ":"
        + preprocessed_scores["POS"].astype(str)
        + ":"
        + preprocessed_scores["EFFECT_ALLELE"]
        + ":"
        + preprocessed_scores["OTHER_ALLELE"]
    )

    preprocessed_scores = preprocessed_scores.set_index("SNPID")
    return preprocessed_scores


def _preprocess_genetic_maps(map_directory_path: str) -> dict[int, list[int]]:
    genetic_maps = _load_genetic_maps_from_feather(map_directory_path)
    bin_mappings = {}
    for i in range(1, 23):
        map_key = f"GeneticMap{i}"
        if map_key in genetic_maps:
            genetic_map = genetic_maps[map_key]
            upper_bounds = genetic_map["upper_bound"].tolist()
            bin_mappings[i] = upper_bounds
    return bin_mappings
