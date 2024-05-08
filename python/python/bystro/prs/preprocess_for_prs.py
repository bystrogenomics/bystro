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
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

AD_SCORE_FILEPATH = "gwas_summary_stats/AD_sumstats_hg38_PMID35379992.feather"


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class GwasStatsLocusKind(StrEnum):
    MATCH = "Direct Match"
    SWAPPED = "Effect Allele Is Ref"
    NO_MATCH = "Alleles Do Not Match"


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


def _extract_nomiss_dosage_loci(dosage_matrix_path: str) -> pd.DataFrame:
    """
    Reads a dataset from the provided dosage matrix Feather file path and filters out
    rows where any column other than 'locus' contains a missing dosage value (represented by -1).

    Args:
    ----
    dosage_matrix_path (str): The path to the Feather file containing the dataset.

    Returns:
    -------
    pd.DataFrame: A DataFrame with a single 'locus' column containing loci with no missing
    dosage values across all other columns.
    """
    dataset = ds.dataset(dosage_matrix_path, format="feather")
    conditions = []
    for column in dataset.schema.names:
        if column != "locus":
            condition = ds.field(column) != -1
            conditions.append(condition)
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition = combined_condition & condition
    filtered_dataset = dataset.filter(combined_condition)
    filtered_loci = filtered_dataset.to_table(columns=["locus"]).to_pandas()
    return filtered_loci


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
    """
    Loads genetic maps from Feather files located in the specified directory.
    For each chromosome, it extracts the upper bound values as lists and
    returns them in a dictionary where the key is the chromosome number.

    Args:
    ----
    map_directory_path (str): Path to the directory containing Feather files for the genetic maps.

    Returns:
    -------
    dict[int, list[int]]: A dictionary where each key is a chromosome number and
    each value is a list of upper bound values extracted from the corresponding genetic map.
    """
    genetic_maps = _load_genetic_maps_from_feather(map_directory_path)
    bin_mappings = {}
    for i in range(1, 23):
        map_key = f"GeneticMap{i}"
        if map_key in genetic_maps:
            genetic_map = genetic_maps[map_key]
            upper_bounds = genetic_map["upper_bound"].tolist()
            bin_mappings[i] = upper_bounds
    return bin_mappings


def read_feather_in_chunks(file_path, columns=None, chunk_size=1000):
    """Read a Feather file in chunks as pandas Dataframes."""
    table = feather.read_table(file_path, columns=columns)
    for i in range(0, table.num_rows, chunk_size):
        chunk = table.slice(i, chunk_size).to_pandas()
        chunk.index = range(i, i + len(chunk))
        yield chunk


def get_p_value_thresholded_indices(df, p_value_threshold):
    """Return indices of rows with P-values less than the specified threshold."""
    return df.index[df["P"] < p_value_threshold].tolist()


def generate_thresholded_overlap_scores_dosage(
    gwas_scores_path: str, dosage_matrix_path: str
) -> pd.DataFrame:
    """Compare and restrict to overlapping loci between dosage matrix and thresholded scores."""
    scores = _preprocess_scores(gwas_scores_path)
    dosage_loci_nomiss = _extract_nomiss_dosage_loci(dosage_matrix_path)

    # TODO: Add customizable p value threshold and option for multiple thresholds
    p_value_threshold = 0.05
    thresholded_indices_set = set()
    for chunk in read_feather_in_chunks(gwas_scores_path, columns=["P"], chunk_size=1000):
        thresholded_indices = get_p_value_thresholded_indices(chunk, p_value_threshold)
        thresholded_indices_set.update(thresholded_indices)
    scores_filtered_indices = sorted(list(thresholded_indices_set))
    thresholded_scores = scores.iloc[scores_filtered_indices]

    set_A = set(thresholded_scores.index)
    set_B = set(dosage_loci_nomiss["locus"])
    overlap_snps = set_A.intersection(set_B)
    remaining_snps_scores = len(thresholded_scores) - len(overlap_snps)
    remaining_snps_frac = (len(thresholded_scores) - len(overlap_snps)) / len(thresholded_scores)
    low_snps_frac = 0.20
    if remaining_snps_frac < low_snps_frac:
        logger.warning
        (
            "Only %s of SNPs out of %s SNPs remain which may limit scoring accuracy",
            remaining_snps_scores,
            len(thresholded_scores),
        )
    scores_overlap = thresholded_scores[thresholded_scores.index.isin(overlap_snps)]
    return scores_overlap


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


def adjust_dosages(row):
    if row["allele_comparison"] == "reversed match":
        for col in row.index:
            if col != "allele_comparison" and row[col] != -1:
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


def assign_bins(scores_overlap: pd.DataFrame, bin_mappings: dict) -> pd.DataFrame:
    """Assign bins to each row in the DataFrame."""
    scores_overlap["bin"] = scores_overlap.apply(find_bin_for_row, bin_mappings=bin_mappings, axis=1)
    return scores_overlap


def calculate_abs_effect_weights(merged_scores_genos: pd.DataFrame) -> pd.DataFrame:
    """Calculate and assign absolute effect weights."""
    merged_scores_genos["abs_effect_weight"] = merged_scores_genos["BETA"].abs()
    return merged_scores_genos


def select_max_effect_per_bin(scores_overlap_abs_val: pd.DataFrame) -> pd.DataFrame:
    """Select the row with the maximum effect weight for each bin."""
    return scores_overlap_abs_val.loc[
        scores_overlap_abs_val.groupby(["CHR", "bin"])["abs_effect_weight"].idxmax()
    ]


def clean_scores_for_analysis(
    max_effect_per_bin: pd.DataFrame, column_to_drop: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop extra columns and prepare df with final set of loci for dosage matrix filtering."""
    scores_overlap_adjusted = max_effect_per_bin.drop(columns=["bin", "abs_effect_weight"])
    scores_overlap_adjusted = scores_overlap_adjusted.set_index("SNPID")
    format_col_index = scores_overlap_adjusted.columns.get_loc(column_to_drop)
    columns_to_keep = ["allele_comparison"] + list(
        scores_overlap_adjusted.columns[format_col_index + 1 :]
    )
    loci_and_allele_comparison = scores_overlap_adjusted[columns_to_keep]
    return scores_overlap_adjusted, loci_and_allele_comparison


def ld_clump(scores_overlap: pd.DataFrame, map_directory_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin using genetic map, clump, and adjust dosages and determine if effect allele is alt or ref."""
    bin_mappings = _preprocess_genetic_maps(map_directory_path)
    scores_overlap = scores_overlap.reset_index()
    allele_comparison_results = scores_overlap.apply(
        compare_alleles, col1="SNPID", col2="ID_effect_as_alt", axis=1
    )
    scores_overlap.insert(0, "allele_comparison", allele_comparison_results)
    scores_overlap_w_bins = assign_bins(scores_overlap, bin_mappings)
    scores_overlap_abs_val = calculate_abs_effect_weights(scores_overlap_w_bins)
    max_effect_per_bin = select_max_effect_per_bin(scores_overlap_abs_val)
    return clean_scores_for_analysis(max_effect_per_bin, "ID_effect_as_ref")


def extract_clumped_thresholded_genos(
    dosage_matrix_path: str, scores_after_c_t: pd.DataFrame
) -> pd.DataFrame:
    dosage_dataset = ds.dataset(dosage_matrix_path, format="feather")
    loci_list = scores_after_c_t.index.tolist()
    loci_filter = ds.field("locus").isin(loci_list)
    filtered_dataset = dosage_dataset.filter(loci_filter)
    filtered_df = filtered_dataset.to_table().to_pandas()
    return filtered_df


def finalize_dosage_scores_after_c_t(
    gwas_scores_path: str, dosage_matrix_path: str, map_directory_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finalize dosage matrix and scores for PRS calculation."""
    scores_overlap = generate_thresholded_overlap_scores_dosage(gwas_scores_path, dosage_matrix_path)
    scores_after_c_t, loci_and_allele_comparison = ld_clump(scores_overlap, map_directory_path)
    dosage_overlap = extract_clumped_thresholded_genos(dosage_matrix_path, scores_after_c_t)
    dosage_overlap = dosage_overlap.set_index("locus")
    merged_allele_comparison_genos = loci_and_allele_comparison.join(dosage_overlap, how="inner")
    merged_allele_comparison_genos = merged_allele_comparison_genos.sort_index()
    scores_after_c_t = scores_after_c_t.sort_index()
    genos_adjusted = merged_allele_comparison_genos.apply(adjust_dosages, axis=1)
    genotypes_adjusted_only = genos_adjusted.iloc[:, 1:]
    genos_transpose = genotypes_adjusted_only.T
    return genos_transpose, scores_after_c_t


def generate_c_and_t_prs_scores(
    gwas_scores_path: str, dosage_matrix_path: str, map_directory_path: str
) -> pd.Series:
    """Calculate PRS."""
    # TODO: Add covariates to model
    genos_transpose, scores_after_c_t = finalize_dosage_scores_after_c_t(
        gwas_scores_path, dosage_matrix_path, map_directory_path
    )
    return genos_transpose @ scores_after_c_t["BETA"]


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
