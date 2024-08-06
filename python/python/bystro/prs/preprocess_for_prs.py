"""
    Process dosage matrix, GWAS summary statistics, and LD maps for PRS C+T calculation.

    Summary statistics are expected to be in the format:
    CHR, POS, OTHER_ALLELE, EFFECT_ALLELE, P, SNPID, BETA, EFFECT_ALLELE_FREQUENCY

    The only required columns are CHR, POS, OTHER_ALLELE, EFFECT_ALLELE, P, SNPID, BETA, EFFECT_ALLELE_FREQUENCY

    SNPID: chr{CHR}:{POS}:{OTHER_ALLELE}:{EFFECT_ALLELE}

    P: p-value of association

    BETA: the log odds ratio (for binary traits) or the regression coefficient (for continuous traits)

    See: PRS_NOTE.pdf for more information on the PRS calculation.
"""

import bisect
from enum import Enum
import logging
from typing import Any, Optional
import os
from bystro.beanstalkd.messages import ProgressReporter
from pandas._libs import missing
import psutil

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
import pyarrow.dataset as ds  # type: ignore
from pyarrow import feather  # type: ignore

from scipy.stats import norm

from bystro.api.auth import CachedAuth
from bystro.utils.covariates import ExperimentMapping
from bystro.proteomics.annotation_interface import get_annotation_result_from_query
from bystro.ancestry.ancestry_types import AncestryResults, PopulationVector, SuperpopVector
from bystro.prs.model import get_sumstats_file, get_map_file
from bystro.utils.timer import Timer

logging.basicConfig(
    filename="preprocess_for_prs.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


pd.options.future.infer_string = True  # type: ignore

HG19_ASSEMBLY = "hg19"
HG38_ASSEMBLY = "hg38"

GENOTYPE_DOSAGE_LOCUS_COLUMN = "locus"
CEU_POP = "CEU"

AFR = "AFR"
AMR = "AMR"
EAS = "EAS"
EUR = "EUR"
SAS = "SAS"
ANCESTRY_SUPERPOPS = [AFR, AMR, EAS, EUR, SAS]

# For now, we will use LD maps from the subpopulations to approximate the ld maps for superpopulations
# for the purposes of calculating the liabilty scale effect Beta for the PRS calculation
SUPERPOP_TO_POP_MAP = {
    AFR: "YRI",
    AMR: "MXL",
    EAS: "JPT",
    EUR: "CEU",
    SAS: "GIH",
}

# hg19 lacks SAS superpopulation, so we will use the mean AF across all populations
# as a proxy for SAS
# this overall AF will be likely larger than the true SAS AF,
# and will therefore lead to likely conservative PRS scores
HG19_GNOMAD_AF_SUPERPOPS = [
    "gnomad.genomes.AF",
    "gnomad.genomes.AF_afr",
    "gnomad.genomes.AF_amr",
    "gnomad.genomes.AF_eas",
    "gnomad.genomes.AF_nfe",
]
HG19_GNOMAD_AF_SUPERPOPS_MAP = {
    "gnomad.genomes.AF": SAS,
    "gnomad.genomes.AF_afr": AFR,
    "gnomad.genomes.AF_amr": AMR,
    "gnomad.genomes.AF_eas": EAS,
    "gnomad.genomes.AF_nfe": EUR,
}
HG38_GNOMAD_AF_SUPERPOPS = [
    "gnomad.genomes.AF_joint_afr",
    "gnomad.genomes.AF_joint_amr",
    "gnomad.genomes.AF_joint_eas",
    "gnomad.genomes.AF_joint_nfe",
    "gnomad.genomes.AF_joint_sas",
]
HG38_GNOMAD_AF_SUPERPOPS_MAP = {
    "gnomad.genomes.AF_joint_afr": AFR,
    "gnomad.genomes.AF_joint_amr": AMR,
    "gnomad.genomes.AF_joint_eas": EAS,
    "gnomad.genomes.AF_joint_nfe": EUR,
    "gnomad.genomes.AF_joint_sas": SAS,
}

######### SUMSTAT COLUMNS #########

CHROM_COLUMN = "CHR"
POS_COLUMN = "POS"
SNPID_COLUMN = "SNPID"
VARIANT_ID_COLUMN = "VARIANT_ID"
OTHER_ALLELE_COLUMN = "OTHER_ALLELE"
EFFECT_ALLELE_COLUMN = "EFFECT_ALLELE"
P_COLUMN = "P"
BETA_COLUMN = "BETA"
BIN_COLUMN = "bin"
EFFECT_FREQUENCY_COLUMN = "EFFECT_ALLELE_FREQUENCY"

HARMONIZED_SUMSTAT_COLUMNS = [
    CHROM_COLUMN,
    POS_COLUMN,
    OTHER_ALLELE_COLUMN,
    EFFECT_ALLELE_COLUMN,
    P_COLUMN,
    BETA_COLUMN,
    EFFECT_FREQUENCY_COLUMN,
    SNPID_COLUMN,
    VARIANT_ID_COLUMN,
]
HARMONIZED_SUMSTAT_COLUMN_TYPES = {
    CHROM_COLUMN: "string[pyarrow]",
    POS_COLUMN: "int64",
    OTHER_ALLELE_COLUMN: "string[pyarrow]",
    EFFECT_ALLELE_COLUMN: "string[pyarrow]",
    P_COLUMN: "float32",
    SNPID_COLUMN: "string[pyarrow]",
    BETA_COLUMN: "float32",
    EFFECT_FREQUENCY_COLUMN: "float32",
}

ID_EFFECT_COMPUTED_COLUMN = "ID_effect_as_alt"
ID_REF_COMPUTED_COLUMN = "ID_effect_as_ref"
ALLELE_COMPARISON_COMPUTED_COLUMN = "allele_comparison"
REVERSE_MATCH_COMPUTED_VALUE = "reversed match"
ABS_EFFECT_COMPUTED_VALUE = "abs_effect_weight"

######### /end SUMSTAT COLUMNS #########

UPPER_BOUND_GENETIC_MAP_COLUMN = "upper_bound"


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

    return ad_scores[HARMONIZED_SUMSTAT_COLUMNS].astype(HARMONIZED_SUMSTAT_COLUMN_TYPES)


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


def _convert_rsid_to_query_format(rsids: set) -> str:
    """
    Convert a set of rsids from the format 'rs123' to
    '(gnomad.genomes.id:rs123)'
    and separate them by '||' in order to issue queries with them.
    """
    if not rsids:
        raise ValueError("No rsids provided for conversion to query format.")

    query = []

    for rsid in rsids:
        single_query = f"(gnomad.genomes.id:{rsid})"
        query.append(single_query)

    return " || ".join(query)


def _extract_af_and_loci_overlap(
    score_loci: set,
    index_name: str,
    assembly: str,
    rsids: set | None = None,
    user: CachedAuth | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Convert loci to query format, perform annotation query,
    and return the loci with gnomad allele frequencies.
    """
    if rsids is not None:
        # check that there are no missing values in the set; by checking for None or np.nan
        if (None in rsids) or (np.nan in rsids):
            logger.warning("There are missing values in the set of rsids, defaulting to score_loci.")
            query = _convert_loci_to_query_format(score_loci)
        else:
            query = _convert_rsid_to_query_format(rsids)
    else:
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
        .set_index(GENOTYPE_DOSAGE_LOCUS_COLUMN)
        .rename(columns=gnomad_af_fields_map)
    )

    return res[ANCESTRY_SUPERPOPS]


def _calculate_ancestry_weighted_af(
    gnomad_afs: pd.DataFrame, superpop_probabilities: dict[str, SuperpopVector]
) -> pd.DataFrame:
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

    sample_superprop_probs_df.to_csv("sample_superprop_probs_df_big_daly.tsv", sep="\t")
    # Normalize, just in case
    sample_superprop_probs_df = sample_superprop_probs_df.div(
        sample_superprop_probs_df.sum(axis=0), axis=1
    )
    sample_superprop_probs_df.to_csv("sample_superprop_probs_df_normalized_big_daly.tsv", sep="\t")

    return gnomad_afs @ sample_superprop_probs_df.loc[gnomad_afs.columns.tolist()]


def _preprocess_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 2 version of the SNPID column, one using the
    effect allele as the alternate allele and one using the
    effect allele as the reference allele. This is to account
    for non-sense strand issues in the GWAS summary statistics.
    """
    preprocessed_scores = scores[HARMONIZED_SUMSTAT_COLUMNS].copy()

    preprocessed_scores[ID_EFFECT_COMPUTED_COLUMN] = (
        "chr"
        + preprocessed_scores[CHROM_COLUMN].apply(str)
        + ":"
        + preprocessed_scores[POS_COLUMN].apply(str)
        + ":"
        + preprocessed_scores[OTHER_ALLELE_COLUMN].apply(str)
        + ":"
        + preprocessed_scores[EFFECT_ALLELE_COLUMN].apply(str)
    )

    preprocessed_scores[ID_REF_COMPUTED_COLUMN] = (
        "chr"
        + preprocessed_scores[CHROM_COLUMN].apply(str)
        + ":"
        + preprocessed_scores[POS_COLUMN].apply(str)
        + ":"
        + preprocessed_scores[EFFECT_ALLELE_COLUMN].apply(str)
        + ":"
        + preprocessed_scores[OTHER_ALLELE_COLUMN].apply(str)
    )

    return preprocessed_scores.set_index(SNPID_COLUMN)


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
            upper_bounds = genetic_map[UPPER_BOUND_GENETIC_MAP_COLUMN].tolist()
            bin_mappings[i] = upper_bounds
    return bin_mappings


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
    if row[ALLELE_COMPARISON_COMPUTED_COLUMN] == REVERSE_MATCH_COMPUTED_VALUE:
        for col in row.index:
            if col != ALLELE_COMPARISON_COMPUTED_COLUMN and row[col] != -1:
                row[col] = 2 - row[col]
    return row


def find_bin_for_row(row: pd.Series, bin_mappings: dict[int, list[int]]) -> Optional[int]:
    """Determine bin for each locus for LD clumping."""
    try:
        chromosome = int(row[CHROM_COLUMN])
    except ValueError:
        return None
    position = row[POS_COLUMN]
    if chromosome in bin_mappings:
        upper_bounds = bin_mappings[chromosome]
        idx = bisect.bisect(upper_bounds, position)
        return idx
    return None


def assign_bins(scores_overlap: pd.DataFrame, bin_mappings: dict) -> pd.DataFrame:
    """Assign bins to each row in the DataFrame."""
    scores_overlap[BIN_COLUMN] = scores_overlap.apply(
        find_bin_for_row, bin_mappings=bin_mappings, axis=1
    )
    return scores_overlap


def calculate_abs_effect_weights(merged_scores_genos: pd.DataFrame) -> pd.DataFrame:
    """Calculate and assign absolute effect weights."""
    merged_scores_genos[ABS_EFFECT_COMPUTED_VALUE] = merged_scores_genos[BETA_COLUMN].abs()
    return merged_scores_genos


def select_max_effect_per_bin(scores_overlap_abs_val: pd.DataFrame) -> pd.DataFrame:
    """Select the row with the maximum effect weight for each bin."""
    return scores_overlap_abs_val.loc[
        scores_overlap_abs_val.groupby([CHROM_COLUMN, BIN_COLUMN])[ABS_EFFECT_COMPUTED_VALUE].idxmax()
    ]


def select_min_pval_per_bin(scores_overlap_abs_val: pd.DataFrame) -> pd.DataFrame:
    """Select the row with the smallest p-value for each bin."""
    return scores_overlap_abs_val.loc[
        scores_overlap_abs_val.groupby([CHROM_COLUMN, BIN_COLUMN])[P_COLUMN].idxmin()
    ]


def clean_scores_for_analysis(
    min_pval_per_bin: pd.DataFrame, column_to_drop: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop extra columns and prepare df with final set of loci for dosage matrix filtering."""
    scores_overlap_adjusted = min_pval_per_bin.drop(columns=[BIN_COLUMN])
    scores_overlap_adjusted = scores_overlap_adjusted.set_index(SNPID_COLUMN)
    format_col_index = scores_overlap_adjusted.columns.get_loc(column_to_drop)
    # If there is more than 1 column found, our "columns_to_keep" function will not work
    if not isinstance(format_col_index, int):
        raise ValueError(f"Column {column_to_drop} was not found uniquely in the dataframe.")
    columns_to_keep = [ALLELE_COMPARISON_COMPUTED_COLUMN] + list(
        scores_overlap_adjusted.columns[format_col_index + 1 :]
    )

    print("columns_to_keep", columns_to_keep)
    loci_and_allele_comparison = scores_overlap_adjusted[columns_to_keep]
    return scores_overlap_adjusted, loci_and_allele_comparison


def ld_clump(scores_overlap: pd.DataFrame, map_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin using genetic map, clump, and adjust dosages and determine if effect allele is alt or ref."""
    bin_mappings = _preprocess_genetic_maps(map_path)
    scores_overlap = scores_overlap.reset_index()
    allele_comparison_results = scores_overlap.apply(
        compare_alleles, col1=SNPID_COLUMN, col2=ID_EFFECT_COMPUTED_COLUMN, axis=1
    )
    scores_overlap.insert(0, ALLELE_COMPARISON_COMPUTED_COLUMN, allele_comparison_results)
    scores_overlap_w_bins = assign_bins(scores_overlap, bin_mappings)

    min_pval_per_bin = select_min_pval_per_bin(scores_overlap_w_bins)
    return clean_scores_for_analysis(min_pval_per_bin, ID_REF_COMPUTED_COLUMN)


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


def get_allelic_effect(df: pd.DataFrame, prevalence: float) -> pd.Series:
    if prevalence <= 0 or prevalence >= 1:
        raise ValueError("Prevalence must be between 0 and 1")

    threshold = norm.ppf(1.0 - prevalence)
    print("threshold", threshold)
    print('df["EFFECT_ALLELE_FREQUENCY"]', df["EFFECT_ALLELE_FREQUENCY"])
    pene_effect = prevalence / (
        df["EFFECT_ALLELE_FREQUENCY"] + (1 - df["EFFECT_ALLELE_FREQUENCY"]) / np.exp(df["BETA"])
    )
    pene_ref = (prevalence - pene_effect * df["EFFECT_ALLELE_FREQUENCY"]) / (
        1 - df["EFFECT_ALLELE_FREQUENCY"]
    )

    print("pene_effect", pene_effect)
    print("pene_ref", pene_ref)

    alpha_effect = threshold - norm.ppf(1.0 - pene_effect)
    alpha_ref = threshold - norm.ppf(1.0 - pene_ref)

    print("alpha_effect", alpha_effect)
    print("alpha_ref", alpha_ref)

    beta = alpha_effect - alpha_ref
    print("beta", beta)

    return beta


def generate_c_and_t_prs_scores(
    assembly: str,
    trait: str,
    pmid: str,
    training_populations: list[str],
    ancestry: AncestryResults,
    dosage_matrix_path: str,
    disease_prevalence: float,
    continuous_trait: bool,
    index_name: str,
    experiment_mapping: ExperimentMapping | None = None,
    min_abs_beta: float = 0.01,
    max_abs_beta: float = 3.0,
    p_value_threshold: float = 0.05,
    sample_chunk_size: int = 200,
    user: CachedAuth | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
    reporter: ProgressReporter | None = None,
) -> pd.DataFrame:
    """
    Calculate PRS scores using the C+T method.

    Parameters
    ----------
    assembly: str
        The genome assembly version.
    trait: str
        The trait name.
    pmid: str
        The PubMed ID of the GWAS study.
    training_populations: list[str]
        The populations or superpopulations from which the training data was derived.
    ancestry: AncestryResults
        The ancestry results for the study.
    dosage_matrix_path: str
        The path to the dosage matrix file.
    experiment_mapping: ExperimentMapping, optional
        The experiment mapping for the study.
    disease_prevalence: float
        The prevalence of the disease.
    continuous_trait: bool
        Whether the trait is continuous.
    p_value_threshold: float, optional
        The p-value threshold for selecting loci. Default is 0.05,
        meaning loci with p-values less than or equal to 0.05 will be selected.
    min_abs_beta: float, optional
        The minimum absolute beta value used in calculating the odds ratio
    max_abs_beta: float, optional
        The maximum absolute beta value used in calculating the odds ratio
    sample_chunk_size: int, optional
        The number of samples to process in each chunk.
    index_name: str
        The index name of the dataset in the OpenSearch cluster.
    cluster_opensearch_config: dict, optional
        The OpenSearch cluster configuration
    reporter: ProgressReporter, optional
        The progress reporter.
    """
    if len(training_populations) > 1:
        raise ValueError(
            (
                "PRS training data from only one superpopulation "
                f"or population is currently supported: found {training_populations}."
            )
        )

    sumstat_population = training_populations[0]

    if sumstat_population in SUPERPOP_TO_POP_MAP:
        sumstat_population = SUPERPOP_TO_POP_MAP[sumstat_population]

    try:
        sumstat_ld_map_path = get_map_file(assembly, sumstat_population)
    except ValueError as e:
        raise ValueError(f"{sumstat_population} is likely not supported. Failed to get map file: {e}")

    if index_name is not None and cluster_opensearch_config is None and user is None:
        raise ValueError(
            "If index_name is provided, either user or cluster_opensearch_config must be provided."
        )

    # This part goes through dosage matrix the first time to get overlapping loci
    with Timer() as timer:
        gwas_scores_path = get_sumstats_file(trait, assembly, pmid)
        scores = _load_association_scores(str(gwas_scores_path))
    logger.debug("Time to load association scores: %s", timer.elapsed_time)

    if reporter is not None:
        reporter.message.remote("Loaded association scores")  # type: ignore

    with Timer() as timer:
        preprocessed_scores = _preprocess_scores(scores)

        # prune preprocessed_scores to only include loci with p-values below the threshold
        # and filter down to sites with beta values within range
        preprocessed_scores = preprocessed_scores[preprocessed_scores[P_COLUMN] <= p_value_threshold]
        preprocessed_scores = preprocessed_scores[
            (preprocessed_scores[BETA_COLUMN].abs() >= min_abs_beta)
            & (preprocessed_scores[BETA_COLUMN].abs() <= max_abs_beta)
        ]

        preprocessed_scores.to_csv("preprocessed_scores_pruned_big_daly.csv")

        scores_after_c_t, _loci_and_allele_comparison = ld_clump(
            preprocessed_scores, str(sumstat_ld_map_path)
        )
        # TODO: check for non-direct match and warn
        preprocessed_scores = preprocessed_scores.loc[scores_after_c_t.index]
    logger.debug("Time to preprocess scores: %s", timer.elapsed_time)

    if reporter is not None:
        reporter.message.remote("Preprocessed scores")  # type: ignore

    preprocessed_scores_loci = set(preprocessed_scores.index)
    score_loci_filter = pc.field(GENOTYPE_DOSAGE_LOCUS_COLUMN).isin(
        pa.array(list(preprocessed_scores_loci))
    )

    dosage_ds = ds.dataset(dosage_matrix_path, format="feather").filter(score_loci_filter)

    dosage_loci = (
        dosage_ds.to_table([GENOTYPE_DOSAGE_LOCUS_COLUMN])
        .to_pandas()[GENOTYPE_DOSAGE_LOCUS_COLUMN]
        .to_numpy()
    )

    assert set(dosage_loci).issubset(preprocessed_scores_loci)

    preprocessed_scores = preprocessed_scores.loc[dosage_loci]

    threshold = norm.ppf(1.0 - disease_prevalence)
    if not continuous_trait:
        preprocessed_scores[BETA_COLUMN] = get_allelic_effect(preprocessed_scores, disease_prevalence)

    preprocessed_scores.to_csv("preprocessed_scores_step3.csv")
    # For now we will ld prune based on a single population, the top_hit
    # To vectorize ld pruning, we will gather chunks based on the top hit

    top_hits, _, superpop_probabilities = _ancestry_results_to_dicts(ancestry)
    sample_groups = []
    population_wise_samples: dict[str, list[str]] = {}
    for sample, population in top_hits.items():
        if population not in population_wise_samples:
            population_wise_samples[population] = []
        population_wise_samples[population].append(sample)

    total_number_of_samples = 0
    for population, samples in population_wise_samples.items():
        for i in range(0, len(samples), sample_chunk_size):
            sample_group = samples[i : i + sample_chunk_size]
            sample_groups.append((population, sample_group))
            total_number_of_samples += len(sample_group)

    mean_q: pd.Series | float = 0.0

    with Timer() as query_timer:
        logger.debug(
            "Memory usage before fetching gnomad allele frequencies: %s",
            psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        )
        population_allele_frequencies = _extract_af_and_loci_overlap(
            score_loci=set(dosage_loci),
            rsids=set(preprocessed_scores[VARIANT_ID_COLUMN]),
            index_name=index_name,
            assembly=assembly,
            user=user,
            cluster_opensearch_config=cluster_opensearch_config,
        )

        logger.debug("population_allele_frequencies: %s", population_allele_frequencies)

        logger.debug(
            "Memory usage after fetching gnomad allele frequencies: %s",
            psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        )

    ancestry_weighted_afs = _calculate_ancestry_weighted_af(
        population_allele_frequencies, superpop_probabilities
    )

    mean_q = ancestry_weighted_afs.fillna(0).mean(axis=1)

    logger.debug("mean_q: %s", mean_q)
    logger.debug("Time to query for gnomad allele frequencies: %s", query_timer.elapsed_time)

    # get all ancestry_weighted_afs loci that are not missing
    dosage_loci_nonmissing_afs = list(
        set(dosage_loci).intersection(set(population_allele_frequencies.index))
    )

    if len(dosage_loci_nonmissing_afs) == 0:
        raise ValueError(
            "No loci match between base and target dataset; cannot proceed with PRS calculation."
        )

    preprocessed_scores = preprocessed_scores.loc[dosage_loci_nonmissing_afs]

    if preprocessed_scores.empty:
        raise ValueError(
            "No loci match between base and target dataset; cannot proceed with PRS calculation."
        )

    or_prev = (1.0 - disease_prevalence) / disease_prevalence

    Va = preprocessed_scores[BETA_COLUMN] * preprocessed_scores[BETA_COLUMN] * 2 * mean_q * (1 - mean_q)
    Va = Va.fillna(0)

    Ve = np.sqrt(1.0 - Va.sum())

    logger.debug("Va %s", Va)
    logger.debug("Ve: %s", Ve)

    ### Debug code to remove
    population_allele_frequencies.to_csv("population_allele.tsv", sep="\t")
    ancestry_weighted_afs.to_csv("ancestry_weighted_afs.tsv", sep="\t")
    mean_q.to_csv("mean_q.tsv", sep="\t")
    preprocessed_scores.to_csv("preprocessed_scores_step4.csv")
    Va.to_csv("Va_big_daly.tsv", sep="\t")
    print("or_prev", or_prev)
    print("scores_overlap[BETA_COLUMN]", preprocessed_scores[BETA_COLUMN].shape)
    print("nan scores_overlap[BETA_COLUMN]", preprocessed_scores[BETA_COLUMN].isna().sum())
    print("mean_q", mean_q)
    print("nan mean_q", mean_q.isna().sum())
    print("scores_overlap[BETA_COLUMN]", preprocessed_scores[BETA_COLUMN])
    print(
        "scores_overlap[BETA_COLUMN]*scores_overlap[BETA_COLUMN]",
        preprocessed_scores[BETA_COLUMN] * preprocessed_scores[BETA_COLUMN],
    )
    ### end Debug code to remove

    if reporter is not None:
        reporter.message.remote("Fetched allele frequencies")  # type: ignore

    # Accumulate the results
    prs_scores: dict[str, float] = {}
    corrected_odds_ratios: dict[str, float] = {}

    dosage_chunk_index: pd.Index = pd.Index(list(dosage_loci_nonmissing_afs), dtype="string[pyarrow]")

    score_loci_filter = pc.field(GENOTYPE_DOSAGE_LOCUS_COLUMN).isin(
        pa.array(list(dosage_loci_nonmissing_afs))
    )
    dosage_ds = ds.dataset(dosage_matrix_path, format="feather").filter(score_loci_filter)
    with Timer() as outer_timer:
        samples_processed = 0

        for sample_pop_group in sample_groups:
            population, sample_group = sample_pop_group

            with Timer() as timer:
                sample_genotypes = dosage_ds.to_table([*sample_group]).to_pandas()
                sample_genotypes.index = dosage_chunk_index
                print(f"sample_genotypes for chunk {samples_processed}", sample_genotypes)

                # TODO @akotlar: 2024-08-06 better imputation strategy
                # Imputation of missing values
                # For now, set missing values to 0, so that they do not affect the PRS calculation
                sample_genotypes = sample_genotypes.fillna(0)  # pre 2024-07 missingness
                sample_genotypes[sample_genotypes < 0] = 0  # post 2024-07 missingness

            logger.debug(
                "Time to load dosage matrix chunk of %d samples (total samples: %d): %s",
                len(sample_group),
                samples_processed + len(sample_group),
                timer.elapsed_time,
            )
            logger.debug(
                "Memory usage after loading dosage matrix of %d samples (total samples: %d): %s",
                len(sample_group),
                samples_processed + len(sample_group),
                psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            )

            missing_loci = set(preprocessed_scores.index) - set(sample_genotypes.index)

            if missing_loci:
                raise ValueError(
                    f"Missing loci in dosage matrix chunk for population {population}: {missing_loci}"
                )

            with Timer() as final_timer:
                genos_transpose = (
                    sample_genotypes.T
                    - 2 * ancestry_weighted_afs.loc[sample_genotypes.index, sample_genotypes.columns].T
                )

                beta_values = scores_after_c_t.loc[list(genos_transpose.columns)][BETA_COLUMN]
                prs_scores_chunk = genos_transpose @ beta_values

                sample_prevalence = 1.0 - norm.cdf(threshold, prs_scores_chunk, Ve)
                real_or = or_prev * sample_prevalence / (1.0 - sample_prevalence)

                # add each prs score to the prs_scores dict
                for index, sample in enumerate(genos_transpose.index):
                    prs_scores[sample] = prs_scores_chunk.loc[sample]
                    corrected_odds_ratios[sample] = real_or[index]

                # debug code to remove
                genos_transpose.to_csv(f"genos_transpose_{samples_processed}.tsv", sep="\t")
                scores_after_c_t.loc[list(genos_transpose.columns)].to_csv(
                    f"scores_after_c_t_{samples_processed}.tsv", sep="\t"
                )
                print("prs_scores_chunk", prs_scores_chunk)
                print("sample_prevalence", sample_prevalence)
                print("real_or", real_or)
                print("corrected_odds_ratios", corrected_odds_ratios)
                # end debug code to remove

            logger.debug(
                "Time to calculate PRS for chunk of %d samples (%d total samples): %s",
                len(sample_group),
                samples_processed + len(sample_group),
                final_timer.elapsed_time,
            )
            logger.debug(
                "Memory usage after calculating PRS for %d samples (%d total samples): %s",
                len(sample_group),
                samples_processed + len(sample_group),
                psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            )

            if reporter is not None:
                samples_processed += len(sample_group)
                reporter.increment_and_write_progress_message.remote(  # type: ignore
                    len(sample_group),
                    "Processed",
                    f"samples ({int((samples_processed/total_number_of_samples) * 10_000)/100}%)",
                    True,
                )
    logger.debug("Time to calculate PRS for all samples: %s", outer_timer.elapsed_time)

    # create dataframe with PRS scores and corrected odds ratios
    results = pd.DataFrame({"PRS": prs_scores, "Corrected OR": corrected_odds_ratios})

    # debug code to remove
    print("results", results)
    # debug code to remove

    return results


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
