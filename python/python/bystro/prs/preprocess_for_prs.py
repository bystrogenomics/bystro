"""Process dosage matrix, GWAS summary statistics, and LD maps for PRS C+T calculation."""

import bisect
import logging
from typing import Optional
import matplotlib.pyplot as plt  # type: ignore
from enum import Enum
import pandas as pd
from pyarrow import feather  # type: ignore
import pyarrow.dataset as ds  # type: ignore
import pyarrow.compute as pc  # type: ignore
import pyarrow as pa  # type: ignore

import numpy as np

from bystro.utils.timer import Timer

logger = logging.getLogger(__name__)

pd.options.future.infer_string = True  # type: ignore


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class GwasStatsLocusKind(StrEnum):
    MATCH = "Direct Match"
    SWAPPED = "Effect Allele Is Ref"
    NO_MATCH = "Alleles Do Not Match"


def _load_association_scores(ad_scores_filepath: str) -> pd.DataFrame:
    """Load in GWAS summary statistics provided by user."""
    # For now, we are supporting one AD dataset PMID:35379992
    ad_scores = feather.read_feather(ad_scores_filepath)

    columns_to_include = ["CHR", "POS", "OTHER_ALLELE", "EFFECT_ALLELE", "P", "SNPID", "BETA"]
    return ad_scores[columns_to_include].astype(
        {
            "CHR": "int64",
            "POS": "int64",
            "OTHER_ALLELE": "string[pyarrow_numpy]",
            "EFFECT_ALLELE": "string[pyarrow_numpy]",
            "P": "float32",
            "SNPID": "string[pyarrow_numpy]",
            "BETA": "float32",
        }
    )


def _load_genetic_maps_from_feather(map_path: str) -> dict[str, pd.DataFrame]:
    """Load genetic maps from Feather files in the specified directory, using logging for messages.

    Args:
    ----
    map_path: The path to the Feather files.

    Returns:
    -------
    A dictionary where keys are 'GeneticMap{chromosome_number}' and values are the corresponding
    DataFrames loaded from Feather files.
    """
    try:
        combined_genetic_map = pd.read_feather(map_path)
        combined_genetic_map = combined_genetic_map.astype(
            {"chromosome_num": "int64", "upper_bound": "int64"}
        )
        logger.info("Successfully loaded combined genetic map from: %s", map_path)
        genetic_maps = {}
        for chrom_num in combined_genetic_map["chromosome_num"].unique():
            chrom_df = combined_genetic_map[combined_genetic_map["chromosome_num"] == chrom_num].copy()
            key = f"GeneticMap{chrom_num}"
            genetic_maps[key] = chrom_df
        return genetic_maps
    except Exception as e:
        logger.exception("Failed to load genetic map from: %s: %s", map_path, e)
        raise e


def _preprocess_scores(ad_scores: pd.DataFrame) -> pd.DataFrame:
    """Process GWAS summary statistics to use effect scores for PRS."""
    # For now, we are supporting one AD dataset PMID:35379992

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
        + preprocessed_scores["CHR"].apply(str)
        + ":"
        + preprocessed_scores["POS"].apply(str)
        + ":"
        + preprocessed_scores["OTHER_ALLELE"].apply(str)
        + ":"
        + preprocessed_scores["EFFECT_ALLELE"].apply(str)
    )

    preprocessed_scores["ID_effect_as_ref"] = (
        "chr"
        + preprocessed_scores["CHR"].apply(str)
        + ":"
        + preprocessed_scores["POS"].apply(str)
        + ":"
        + preprocessed_scores["EFFECT_ALLELE"].apply(str)
        + ":"
        + preprocessed_scores["OTHER_ALLELE"].apply(str)
    )
    return preprocessed_scores.set_index("SNPID")


def _preprocess_genetic_maps(map_path: str) -> dict[int, list[int]]:
    """
    Loads genetic maps from Feather files located in the specified directory.
    For each chromosome, it extracts the upper bound values as lists and
    returns them in a dictionary where the key is the chromosome number.

    Args:
    ----
    map_path (str): Path to the directory containing Feather files for the genetic maps.

    Returns:
    -------
    dict[int, list[int]]: A dictionary where each key is a chromosome number and
    each value is a list of upper bound values extracted from the corresponding genetic map.
    """
    genetic_maps = _load_genetic_maps_from_feather(map_path)
    bin_mappings = {}
    for i in range(1, 23):
        map_key = f"GeneticMap{i}"
        if map_key in genetic_maps:
            genetic_map = genetic_maps[map_key]
            upper_bounds = genetic_map["upper_bound"].tolist()
            bin_mappings[i] = upper_bounds
    return bin_mappings


def get_p_value_thresholded_indices(df, p_value_threshold: float) -> set:
    """Return indices of rows with P-values less than the specified threshold."""
    if not (0 <= p_value_threshold <= 1):
        raise ValueError("p_value_threshold must be between 0 and 1")
    return set(df.index[df["P"] < p_value_threshold])


def generate_overlap_scores_dosage(thresholded_score_loci: set, filtered_dosage_loci: set) -> set:
    """Compare and restrict to overlapping loci between dosage matrix and thresholded scores."""

    overlap_loci = thresholded_score_loci.intersection(filtered_dosage_loci)
    if len(overlap_loci) == 0:
        raise ValueError(
            "No loci match between base and target dataset; cannot proceed with PRS calculation."
        )
    return overlap_loci


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
    # If there is more than 1 column found, our "columns_to_keep" function will not work
    if not isinstance(format_col_index, int):
        raise ValueError(f"Column {column_to_drop} was not found uniquely in the dataframe.")
    columns_to_keep = ["allele_comparison"] + list(
        scores_overlap_adjusted.columns[format_col_index + 1 :]
    )
    loci_and_allele_comparison = scores_overlap_adjusted[columns_to_keep]
    return scores_overlap_adjusted, loci_and_allele_comparison


def ld_clump(scores_overlap: pd.DataFrame, map_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin using genetic map, clump, and adjust dosages and determine if effect allele is alt or ref."""
    bin_mappings = _preprocess_genetic_maps(map_path)
    scores_overlap = scores_overlap.reset_index()
    allele_comparison_results = scores_overlap.apply(
        compare_alleles, col1="SNPID", col2="ID_effect_as_alt", axis=1
    )
    scores_overlap.insert(0, "allele_comparison", allele_comparison_results)
    scores_overlap_w_bins = assign_bins(scores_overlap, bin_mappings)
    scores_overlap_abs_val = calculate_abs_effect_weights(scores_overlap_w_bins)
    max_effect_per_bin = select_max_effect_per_bin(scores_overlap_abs_val)
    return clean_scores_for_analysis(max_effect_per_bin, "ID_effect_as_ref")


def finalize_scores_after_c_t(
    gwas_scores_path: str, map_path: str, p_value_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finalize scores for PRS calculation."""
    with Timer() as timer:
        scores = _load_association_scores(gwas_scores_path)
    logger.debug("Time to load association scores: %s", timer.elapsed_time)

    with Timer() as timer:
        preprocessed_scores = _preprocess_scores(scores)
    logger.debug("Time to preprocess scores: %s", timer.elapsed_time)

    with Timer() as timer:
        thresholded_score_loci = get_p_value_thresholded_indices(preprocessed_scores, p_value_threshold)
    logger.debug("Time to get p-value thresholded indices: %s", timer.elapsed_time)

    with Timer() as timer:
        scores_overlap = preprocessed_scores[preprocessed_scores.index.isin(thresholded_score_loci)]
        scores_after_c_t, loci_and_allele_comparison = ld_clump(scores_overlap, map_path)
        scores_after_c_t = scores_after_c_t.sort_index()
    logger.debug("Time to clump and adjust dosages: %s", timer.elapsed_time)

    return scores_after_c_t, loci_and_allele_comparison


def finalize_dosage_after_c_t(
    dosage_overlap: pd.DataFrame, loci_and_allele_comparison: pd.DataFrame
) -> pd.DataFrame:
    """Finalize dosage matrix for PRS calculation."""
    merged_allele_comparison_genos = loci_and_allele_comparison.join(dosage_overlap, how="inner")
    merged_allele_comparison_genos = merged_allele_comparison_genos.sort_index()
    genos_adjusted = merged_allele_comparison_genos.apply(adjust_dosages, axis=1)
    genotypes_adjusted_only = genos_adjusted.iloc[:, 1:]
    genos_transpose = genotypes_adjusted_only.T
    return genos_transpose


def generate_c_and_t_prs_scores(
    gwas_scores_path: str,
    dosage_matrix_path: str,
    map_path: str,
    p_value_threshold: float = 0.05,
) -> pd.Series:
    """Calculate PRS."""
    # This part goes through dosage matrix the first time to get overlapping loci
    with Timer() as timer:
        scores_after_c_t, loci_and_allele_comparison = finalize_scores_after_c_t(
            gwas_scores_path, map_path, p_value_threshold
        )
    logger.debug("Time to finalize scores after clumping and thresholding: %s", timer.elapsed_time)

    # This part goes through dosage matrix the second time, adjusts dosages,
    # transposes genotypes, and calculates PRS
    prs_scores: pd.Series = pd.Series(dtype=np.float32, name="PRS")

    beta_values = scores_after_c_t["BETA"]
    finalized_loci = scores_after_c_t.index.tolist()

    score_loci_filter = pc.field("locus").isin(pa.array(finalized_loci))

    dosage_ds = ds.dataset(dosage_matrix_path, format="feather").filter(score_loci_filter)

    samples = [name for name in dosage_ds.schema.names if name != "locus"]
    sample_groups = [samples[i : i + 1000] for i in range(0, len(samples), 1000)]

    with Timer() as outer_timer:
        for sample_group in sample_groups:
            with Timer() as timer:
                sample_genotypes = dosage_ds.to_table(["locus", *sample_group]).to_pandas()
                sample_genotypes = sample_genotypes.set_index("locus")
                sample_genotypes = sample_genotypes[
                    sample_genotypes.notna() & (sample_genotypes >= 0).all(axis=1)
                ]

            logger.debug(
                "Time to load dosage matrix chunk of %d samples: %s",
                len(sample_group),
                timer.elapsed_time,
            )

            with Timer() as timer:
                genos_transpose = finalize_dosage_after_c_t(sample_genotypes, loci_and_allele_comparison)
                prs_scores_chunk = genos_transpose @ beta_values.loc[genos_transpose.columns]
                prs_scores = prs_scores.add(prs_scores_chunk, fill_value=0)
            logger.debug(
                "Time to calculate PRS for chunk of %d samples: %s",
                len(sample_group),
                timer.elapsed_time,
            )
    logger.debug("Time to calculate PRS for all samples: %s", outer_timer.elapsed_time)

    return prs_scores


def prs_histogram(
    prs_scores: dict,
    bins: Optional[int] = None,
    color: str = "blue",
    title: str = "Histogram of PRS Scores",
    xlabel: str = "PRS Score",
    ylabel: str = "Frequency",
) -> None:
    """Plot for PRS score overview."""
    prs_scores_list = list(prs_scores.values())
    if bins is None:
        bins = int(len(prs_scores_list) ** 0.5) + 1
    plt.figure(figsize=(10, 6))
    plt.hist(prs_scores_list, bins=bins, color=color, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def _load_covariates(COVARIATE_FILEPATH: str) -> pd.DataFrame:
    return pd.read_csv(COVARIATE_FILEPATH, delimiter="\t")
