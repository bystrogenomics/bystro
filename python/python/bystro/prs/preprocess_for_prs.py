"""Process dosage matrix, GWAS summary statistics, and LD maps for PRS C+T calculation."""

import bisect
from enum import Enum
import logging
from typing import Any, Optional
import os
import psutil

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
from pyarrow import feather  # type: ignore
import pyarrow.dataset as ds  # type: ignore
import pyarrow.compute as pc  # type: ignore
import pyarrow as pa  # type: ignore

import numpy as np

from bystro.api.auth import CachedAuth
from bystro.proteomics.annotation_interface import get_annotation_result_from_query
from bystro.ancestry.ancestry_types import AncestryResults, PopulationVector, SuperpopVector
from bystro.prs.model import get_sumstats_file, get_map_file
from bystro.utils.timer import Timer

logger = logging.getLogger(__name__)

pd.options.future.infer_string = True  # type: ignore

HG19_ASSEMBLY = "hg19"
HG38_ASSEMBLY = "hg38"

ANCESTRY_SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
HG19_GNOMAD_AF_SUPERPOPS = [
    "gnomad.genomes.AF_afr",
    "gnomad.genomes.AF_amr",
    "gnomad.genomes.AF_eas",
    "gnomad.genomes.AF_nfe",
]
HG19_GNOMAD_AF_SUPERPOPS_MAP = {
    "gnomad.genomes.AF_afr": "AFR",
    "gnomad.genomes.AF_amr": "AMR",
    "gnomad.genomes.AF_eas": "EAS",
    "gnomad.genomes.AF_nfe": "EUR",
}
HG38_GNOMAD_AF_SUPERPOPS = [
    "gnomad.genomes.AF_joint_afr",
    "gnomad.genomes.AF_joint_amr",
    "gnomad.genomes.AF_joint_eas",
    "gnomad.genomes.AF_joint_nfe",
    "gnomad.genomes.AF_joint_sas",
]
HG38_GNOMAD_AF_SUPERPOPS_MAP = {
    "gnomad.genomes.AF_joint_afr": "AFR",
    "gnomad.genomes.AF_joint_amr": "AMR",
    "gnomad.genomes.AF_joint_eas": "EAS",
    "gnomad.genomes.AF_joint_nfe": "EUR",
    "gnomad.genomes.AF_joint_sas": "SAS",
}


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


def _convert_loci_to_query_format(score_loci: set) -> str:
    """
    Convert a set of loci from the format 'chr10:105612479:G:T' to
    '(chrom:chr10 pos:105612479 inputRef:G alt:T)'
    and separate them by '||' in order to issue queries with them.
    """
    if not score_loci:
        raise ValueError("No loci provided for conversion to query format.")

    finalized_loci = []
    for locus in score_loci:
        chrom, pos, inputRef, alt = locus.split(":")
        single_query = f"(chrom:{chrom} pos:{pos} inputRef:{inputRef} alt:{alt})"
        finalized_loci.append(single_query)

    return f"(_exists_:gnomad.genomes) && ({' || '.join(finalized_loci)})"


def _add_sas_column_if_missing(df):
    if "SAS" not in df.columns:
        df["SAS"] = 0
    return df


def _extract_af_and_loci_overlap(
    score_loci: set,
    index_name: str,
    assembly: str,
    user: CachedAuth | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Convert loci to query format, perform annotation query,
    and return the loci with gnomad allele frequencies.
    """
    query = _convert_loci_to_query_format(score_loci)

    if assembly == HG19_ASSEMBLY:
        gnomad_af_fields = HG19_GNOMAD_AF_SUPERPOPS
        gnomad_af_fields_map = HG19_GNOMAD_AF_SUPERPOPS_MAP
    elif assembly == HG38_ASSEMBLY:
        gnomad_af_fields = HG38_GNOMAD_AF_SUPERPOPS
        gnomad_af_fields_map = HG38_GNOMAD_AF_SUPERPOPS_MAP
    else:
        raise ValueError(f"Assembly {assembly} is not supported.")

    res = (
        get_annotation_result_from_query(
            query_string=query,
            index_name=index_name,
            bystro_api_auth=user,
            cluster_opensearch_config=cluster_opensearch_config,
            melt_samples=False,
            fields=gnomad_af_fields,
        )
        .set_index("locus")
        .fillna(0)
        .rename(columns=gnomad_af_fields_map)
    )

    if assembly == HG19_ASSEMBLY:
        return _add_sas_column_if_missing(res)[ANCESTRY_SUPERPOPS]

    return res[ANCESTRY_SUPERPOPS]


def _calculate_allele_frequency_total_variation(
    gnomad_afs: pd.DataFrame, superpop_probabilities: dict[str, SuperpopVector]
):
    # Renormalize gnomad allele frequencies
    gnomad_afs = gnomad_afs.div(gnomad_afs.sum(axis=1), axis=0)

    sample_superprop_probs: dict[str, dict[str, float]] = {}
    for sample in superpop_probabilities:
        sample_superprop_probs[sample] = {}
        for superpop in ANCESTRY_SUPERPOPS:
            one_sample = superpop_probabilities[sample]
            prob_range = getattr(one_sample, superpop)
            sample_superprop_probs[sample][superpop] = (
                prob_range.lower_bound + prob_range.upper_bound
            ) / 2

    sample_superprop_probs_df = pd.DataFrame(sample_superprop_probs, dtype=np.float32)
    # Normalize, just in case
    sample_superprop_probs_df = sample_superprop_probs_df.div(
        sample_superprop_probs_df.sum(axis=0), axis=1
    )

    return 2 * (gnomad_afs @ sample_superprop_probs_df.loc[gnomad_afs.columns.tolist()])


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


def _ancestry_results_to_dicts(
    ancestry: AncestryResults,
) -> tuple[dict[str, str], dict[str, PopulationVector], dict[str, SuperpopVector]]:
    """Convert AncestryResults to a dictionary of top hit probabilities."""
    top_hits = {}
    populations: dict[str, PopulationVector] = {}
    superpops: dict[str, SuperpopVector] = {}
    for result in ancestry.results:
        top_hits[result.sample_id] = result.top_hit.populations[0]
        populations[result.sample_id] = result.populations
        superpops[result.sample_id] = result.superpops

    return top_hits, populations, superpops


def generate_c_and_t_prs_scores(
    assembly: str,
    trait: str,
    pmid: str,
    ancestry: AncestryResults,
    dosage_matrix_path: str,
    p_value_threshold: float = 0.05,
    sample_chunk_size: int = 200,
    index_name: str | None = None,
    user: CachedAuth | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
) -> pd.Series:
    """Calculate PRS."""
    if index_name is not None and cluster_opensearch_config is None and user is None:
        raise ValueError(
            "If index_name is provided, either user or cluster_opensearch_config must be provided."
        )

    # This part goes through dosage matrix the first time to get overlapping loci
    with Timer() as timer:
        gwas_scores_path = get_sumstats_file(trait, assembly, pmid)
        scores = _load_association_scores(str(gwas_scores_path))
    logger.debug("Time to load association scores: %s", timer.elapsed_time)

    with Timer() as timer:
        preprocessed_scores = _preprocess_scores(scores)
        thresholded_score_loci = get_p_value_thresholded_indices(preprocessed_scores, p_value_threshold)
    logger.debug("Time to preprocess scores: %s", timer.elapsed_time)

    score_loci_filter = pc.field("locus").isin(pa.array(thresholded_score_loci))

    dosage_ds = ds.dataset(dosage_matrix_path, format="feather").filter(score_loci_filter)

    # For now we will ld prune based on a single population, the top_hit
    # To vectorize ld pruning, we will gather chunks based on the top hit

    top_hits, _, superpop_probabilities = _ancestry_results_to_dicts(ancestry)
    sample_groups = []
    population_wise_samples: dict[str, list[str]] = {}
    for sample, population in top_hits.items():
        if population not in population_wise_samples:
            population_wise_samples[population] = []
        population_wise_samples[population].append(sample)

    for population, samples in population_wise_samples.items():
        for i in range(0, len(samples), sample_chunk_size):
            sample_groups.append((population, samples[i : i + sample_chunk_size]))

    ancestry_weighted_af_total_variation: pd.DataFrame | None = None
    if index_name is not None:
        with Timer() as query_timer:
            process = psutil.Process(os.getpid())
            logger.debug(
                "Memory usage before fetching gnomad allele frequencies: %s",
                process.memory_info().rss / 1e6,
            )
            thresholded_loci_gnomad_afs = _extract_af_and_loci_overlap(
                score_loci=thresholded_score_loci,
                index_name=index_name,
                assembly=assembly,
                user=user,
                cluster_opensearch_config=cluster_opensearch_config,
            )

            logger.debug("thresholded_loci_gnomad_afs: %s", thresholded_loci_gnomad_afs)

            ancestry_weighted_af_total_variation = _calculate_allele_frequency_total_variation(
                thresholded_loci_gnomad_afs, superpop_probabilities
            )

            logger.debug(
                "ancestry_weighted_af_total_variation: %s", ancestry_weighted_af_total_variation
            )

            logger.debug(
                "Memory usage after fetching gnomad allele frequencies: %s",
                process.memory_info().rss / 1e6,
            )
        logger.debug("Time to query for gnomad allele frequencies: %s", query_timer.elapsed_time)

    # Accumulate the results
    prs_scores: pd.Series = pd.Series(dtype=np.float32, name="PRS")
    with Timer() as outer_timer:
        for sample_pop_group in sample_groups:
            population, sample_group = sample_pop_group

            with Timer() as timer:
                sample_genotypes = dosage_ds.to_table(["locus", *sample_group]).to_pandas()
                sample_genotypes = sample_genotypes.set_index("locus")

                mask = sample_genotypes.notna().all(axis=1) & (sample_genotypes >= 0).all(axis=1)
                sample_genotypes = sample_genotypes[mask]

                dosage_loci_nomiss = sample_genotypes.index.tolist()

                overlap_loci = generate_overlap_scores_dosage(thresholded_score_loci, dosage_loci_nomiss)
                scores_overlap = preprocessed_scores[preprocessed_scores.index.isin(overlap_loci)]

                # TODO 2024-06-22 @ctrevino: We do not currently have all 26 1000G population maps
                # this block needs to be removed once we do
                try:
                    map_path = get_map_file(assembly, population)
                except ValueError as e:
                    logger.exception(
                        "Failed to get map file for %s: %s, defaulting to CEU", population, e
                    )
                    map_path = get_map_file(assembly, "CEU")

                scores_after_c_t, loci_and_allele_comparison = ld_clump(scores_overlap, str(map_path))
                scores_after_c_t = scores_after_c_t.sort_index()

                beta_values = scores_after_c_t["BETA"]

                with Timer() as timer:
                    genos_transpose = finalize_dosage_after_c_t(
                        sample_genotypes, loci_and_allele_comparison
                    )

                    if ancestry_weighted_af_total_variation is not None:
                        weights_filtered = ancestry_weighted_af_total_variation.loc[
                            genos_transpose.columns
                        ].fillna(0)
                        genos_transpose = genos_transpose - weights_filtered[genos_transpose.index].T

                    prs_scores_chunk = genos_transpose @ beta_values.loc[genos_transpose.columns]
                    prs_scores = prs_scores.add(prs_scores_chunk, fill_value=0)
                logger.debug(
                    "Time to calculate PRS for chunk of %d samples: %s",
                    len(sample_group),
                    timer.elapsed_time,
                )
            logger.debug(
                "Time to load dosage matrix chunk of %d samples: %s",
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
