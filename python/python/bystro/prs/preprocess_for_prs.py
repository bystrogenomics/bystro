"""Process dosage matrix, GWAS summary statistics, and LD maps for PRS C+T calculation."""

import bisect
import glob
import logging
import re
from typing import Optional, Union
import matplotlib.pyplot as plt  # type: ignore
from enum import Enum
import pandas as pd
from pyarrow import feather  # type: ignore

logger = logging.getLogger(__name__)

AD_SCORE_FILEPATH = "gwas_summary_stats/AD_sumstats_PMID35379992.feather"

class StrEnum(str, Enum):
    def __str__(self):
        return self.value

class GwasStatsLocusKind(StrEnum):
    MATCH = "Direct Match"
    SWAPPED = "Effect Allele Is Ref"
    NO_MATCH = "Alleles Do Not Match"


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


def filter_scores_by_p_value(scores: pd.DataFrame, p_value_threshold: float) -> pd.DataFrame:
    """Filter to keep rows with P-values less than the specified threshold for C+T method."""
    return scores[scores["P"] < p_value_threshold]


def generate_thresholded_overlap_scores_dosage(
    gwas_scores_path: str, dosage_matrix_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare and restrict to overlapping loci between dosage matrix and scores."""
    scores = _preprocess_scores(gwas_scores_path)
    dosage_feather = _load_preprocessed_dosage_matrix(dosage_matrix_path)
    # TODO: Add customizable p value threshold and option for multiple thresholds
    thresholded_scores = filter_scores_by_p_value(scores, 0.05)
    set_A = set(thresholded_scores.index)
    set_B = set(dosage_feather["locus"])
    overlap_snps = set_A.intersection(set_B)
    remaining_snps_scores = len(thresholded_scores) - len(overlap_snps)
    remaining_snps_frac = (len(thresholded_scores) - len(overlap_snps)) / len(thresholded_scores)
    low_snps_frac = 0.20
    if remaining_snps_frac < low_snps_frac:
        logger.warning
        (
            "Only %s of SNPs out of %s SNPs remain which may limit scoring calculations",
            remaining_snps_scores,
            len(thresholded_scores),
        )
    scores_overlap = thresholded_scores[thresholded_scores.index.isin(overlap_snps)]
    fulldosagevcf_overlap = dosage_feather[dosage_feather["locus"].isin(overlap_snps)]
    fulldosagevcf_overlap = fulldosagevcf_overlap.set_index("locus")
    merged_scores_genos = scores_overlap.join(fulldosagevcf_overlap, how="inner")
    merged_scores_genos = merged_scores_genos.reset_index()
    return merged_scores_genos, scores_overlap


def compare_alleles(row: pd.Series, col1: str, col2: str) -> GwasStatsLocusKind:
    """Describe whether reference and alternate alleles match effect and other alleles.
    Requires GWAS summary stat SNP IDs and dosage matrix loci to be on the same strand."""
    id1, id2 = row[col1], row[col2]
    chrom1, pos1, ref1, alt1 = id1.split(":")
    chrom2, pos2, ref2, alt2 = id2.split(":")
    if (chrom1, pos1, ref1, alt1) == (chrom2, pos2, ref2, alt2):
        return GwasStatsLocusKind.MATCH
    if (chrom1, pos1, alt1, ref1) == (chrom2, pos2, ref2, alt2):
        return GwasStatsLocusKind.SWAPPED
    return GwasStatsLocusKind.NO_MATCH


def adjust_dosages(row: pd.Series) -> pd.Series:
    """Adjust the dosage for if the effect allele is not the alternate allele."""
    if row["allele_comparison"] == "effect allele is ref":
        for col in row.index:
            if col != "allele_comparison":
                row[col] = 2 - row[col]
    return row


def find_bin_for_row(row: pd.Series, bin_mappings: dict[int, list[int]]) -> Optional[int]:
    """Determine bin for each locus for LD clumping."""
    try:
        chromosome = int(row["CHR"])
    except ValueError:
        return None
    position = row["POS"]
    if chromosome in bin_mappings:
        upper_bounds = bin_mappings[chromosome]
        idx = bisect.bisect(upper_bounds, position)
        return idx
    return None


def assign_bins(merged_scores_genos: pd.DataFrame, bin_mappings: dict) -> pd.DataFrame:
    """Assign bins to each row in the DataFrame."""
    merged_scores_genos["bin"] = merged_scores_genos.apply(
        find_bin_for_row, bin_mappings=bin_mappings, axis=1
    )
    return merged_scores_genos


def calculate_abs_effect_weights(merged_scores_genos: pd.DataFrame) -> pd.DataFrame:
    """Calculate and assign absolute effect weights."""
    merged_scores_genos["abs_effect_weight"] = merged_scores_genos["BETA"].abs()
    return merged_scores_genos


def select_max_effect_per_bin(merged_scores_genos: pd.DataFrame) -> pd.DataFrame:
    """Select the row with the maximum effect weight for each bin."""
    return merged_scores_genos.loc[
        merged_scores_genos.groupby(["CHR", "bin"])["abs_effect_weight"].idxmax()
    ]


def clean_dosage_for_analysis(merged_scores_genos: pd.DataFrame, column_to_drop: str) -> pd.DataFrame:
    """Drop missing values and adjust scores."""
    column_to_drop_index = merged_scores_genos.columns.get_loc(column_to_drop)
    columns_to_keep = [
        "allele_comparison",
        *list(merged_scores_genos.columns[column_to_drop_index + 1 :]),
    ]
    genos_wo_extra_cols = merged_scores_genos[columns_to_keep]
    genos_wo_missing = genos_wo_extra_cols.dropna()
    allele_comparison_results = genos_wo_missing.apply(
        compare_alleles, col1="index", col2="ID_effect_as_alt", axis=1
    )
    genos_adjusted = allele_comparison_results.apply(adjust_dosages, axis=1)
    genos_wo_nomatch_alleles = genos_adjusted[
        genos_adjusted["allele_comparison"] != "alleles do not match"
    ]
    return genos_wo_nomatch_alleles.iloc[:, 1:]


def ld_clump(merged_scores_genos: pd.DataFrame, map_directory_path: str) -> pd.DataFrame:
    """Bin using genetic map, clump, and adjust dosages."""
    bin_mappings = _preprocess_genetic_maps(map_directory_path)
    merged_scores_genos_w_bins = assign_bins(merged_scores_genos, bin_mappings)
    merged_scores_genos_abs_val = calculate_abs_effect_weights(merged_scores_genos_w_bins)
    max_effect_per_bin = select_max_effect_per_bin(merged_scores_genos_abs_val)
    return clean_dosage_for_analysis(max_effect_per_bin, "ID_effect_as_ref")


def finalize_dosage_scores_after_c_t(
    gwas_scores_path: str, dosage_matrix_path: str, map_directory_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finalize dosage matrix and scores for PRS calculation."""
    merged_scores_genos, scores_overlap = generate_thresholded_overlap_scores_dosage(
        gwas_scores_path, dosage_matrix_path
    )
    genotypes_clumped = ld_clump(merged_scores_genos, map_directory_path)
    scores_overlap_adjusted = scores_overlap[scores_overlap.index.isin(genotypes_clumped.index)]
    scores_overlap_adjusted = scores_overlap_adjusted.sort_index()
    genos_transpose = genotypes_clumped.T
    return genos_transpose, scores_overlap_adjusted


def generate_c_and_t_prs_scores(
    gwas_scores_path: str, dosage_matrix_path: str, map_directory_path: str
) -> pd.Series:
    """Calculate PRS."""
    # TODO: Add covariates to model
    genos_transpose, scores_overlap_adjusted = finalize_dosage_scores_after_c_t(
        gwas_scores_path, dosage_matrix_path, map_directory_path
    )
    return genos_transpose @ scores_overlap_adjusted["BETA"]


def prs_histogram(
    prs_scores: Union[list[float], pd.Series],
    bins: int = 20,
    color: str = "blue",
    title: str = "Histogram of PRS Scores",
    xlabel: str = "PRS Score",
    ylabel: str = "Frequency",
) -> None:
    """Plot for PRS score overview."""
    plt.figure(figsize=(10, 6))
    plt.hist(prs_scores, bins=bins, color=color, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def _load_covariates(COVARIATE_FILEPATH: str) -> pd.DataFrame:
    return pd.read_csv(COVARIATE_FILEPATH, delimiter="\t")
