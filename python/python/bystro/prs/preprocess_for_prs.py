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


def prune_by_window(df: pd.DataFrame, window_size: int):
    # Split the SNPID column into separate chromosome and position columns
    df[["Chromosome", "Position", "Ref", "Alt"]] = df[SNPID_COLUMN].str.split(":", expand=True)

    # Convert the Position column to integers
    df["Position"] = df["Position"].astype(int)

    # Sort by Chromosome and Position
    df = df.sort_values(by=["Chromosome", "Position"]).reset_index(drop=True)

    # Function to perform binning and pruning
    def bin_and_prune(df):
        # Initialize the bin column
        df["Bin"] = -1

        # Assign bins based on the position
        bin_counter = 0
        for chrom in df["Chromosome"].unique():
            chrom_df = df[df["Chromosome"] == chrom]
            last_position = (
                -window_size
            )  # Initialize to a negative value to ensure the first SNP is in the first bin
            for idx, row in chrom_df.iterrows():
                if row["Position"] - last_position >= window_size:
                    bin_counter += 1
                    last_position = row["Position"]
                df.at[idx, "Bin"] = bin_counter

        # Within each bin, keep the SNP with the lowest p-value
        pruned_df = df.loc[df.groupby("Bin")[P_COLUMN].idxmin()].reset_index(drop=True)

        # Drop the intermediate columns if not needed
        pruned_df = pruned_df.drop(columns=["Bin"])

        return pruned_df

    # First pass of binning and pruning
    pruned_df = bin_and_prune(df)

    # Second pass of binning and pruning to ensure no overlapping windows
    final_pruned_df = bin_and_prune(pruned_df)

    # Drop the intermediate columns if not needed
    return final_pruned_df.drop(columns=["Chromosome", "Position", "Ref", "Alt"])


# for debug testing against big_daly
_cutler_snp_set = set(
    [
        "rs11121129",
        "rs61839660",
        "rs2256028",
        "rs2697290",
        "rs2282978",
        "rs7236492",
        "rs144483537",
        "rs11793497",
        "rs230261",
        "rs35695082",
        "rs2283789",
        "rs3130275",
        "rs10177855",
        "rs9257248",
        "rs11677053",
        "rs1953970",
        "rs6931409",
        "rs76009660",
        "rs79493594",
        "rs1875124",
        "rs7800747",
        "rs12131796",
        "rs1035441",
        "rs11579874",
        "rs1240708",
        "rs6888346",
        "rs11716895",
        "rs9257828",
        "rs2243288",
        "rs2666986",
        "rs495922",
        "rs727563",
        "rs7733977",
        "rs1080241",
        "rs661356",
        "rs1876905",
        "rs1990760",
        "rs75774207",
        "rs10484439",
        "rs1570694",
        "rs1061758",
        "rs11585739",
        "rs2074023",
        "rs12712063",
        "rs1990623",
        "rs4869311",
        "rs9275602",
        "rs11187157",
        "rs67060340",
        "rs1059823",
        "rs281423",
        "rs11582349",
        "rs9262499",
        "rs267265",
        "rs9815292",
        "rs7097656",
        "rs6011182",
        "rs1736142",
        "rs3136534",
        "rs6500291",
        "rs11709546",
        "rs12215079",
        "rs10445308",
        "rs13218875",
        "rs7611164",
        "rs17772583",
        "rs2690110",
        "rs2153977",
        "rs11743851",
        "rs2071984",
        "rs12411259",
        "rs3130834",
        "rs9387010",
        "rs2503322",
        "rs7566220",
        "rs68191",
        "rs1535039",
        "rs113741533",
        "rs12194935",
        "rs3935123",
        "rs10182181",
        "rs1965582",
        "rs34804116",
        "rs7070722",
        "rs2071025",
        "rs3008855",
        "rs6738394",
        "rs4712527",
        "rs422562",
        "rs2188958",
        "rs7159324",
        "rs76546301",
        "rs11713694",
        "rs2872979",
        "rs241449",
        "rs2486484",
        "rs2978826",
        "rs17145738",
        "rs17718834",
        "rs1859962",
        "rs7015630",
        "rs10411210",
        "rs2746150",
        "rs533852",
        "rs62057818",
        "rs2488401",
        "rs112578597",
        "rs80099993",
        "rs11167518",
        "rs1472971",
        "rs77767934",
        "rs4452638",
        "rs9848203",
        "rs104895045",
        "rs56163845",
        "rs6651252",
        "rs4704727",
        "rs2073167",
        "rs1177266",
        "rs9285919",
        "rs2236974",
        "rs9263600",
        "rs35285054",
        "rs28370830",
        "rs12060309",
        "rs13212651",
        "rs2143950",
        "rs13006559",
        "rs13219140",
        "rs9424945",
        "rs1525235",
        "rs75542709",
        "rs4896243",
        "rs10783854",
        "rs1535",
        "rs17694108",
        "rs11676348",
        "rs6914787",
        "rs404240",
        "rs74869386",
        "rs3024505",
        "rs6711874",
        "rs10923931",
        "rs116149309",
        "rs73078304",
        "rs9868633",
        "rs727330",
        "rs4737089",
        "rs3130662",
        "rs4719613",
        "rs298259",
        "rs7587725",
        "rs16868789",
        "rs7528377",
        "rs724016",
        "rs3807039",
        "rs4957341",
        "rs1545255",
        "rs12968499",
        "rs17227589",
        "rs11752561",
        "rs231764",
        "rs78447894",
        "rs17293632",
        "rs4142219",
        "rs72812826",
        "rs190246",
        "rs4541435",
        "rs3731570",
        "rs2284553",
        "rs3132389",
        "rs2334255",
        "rs2021716",
        "rs6738490",
        "rs177992",
        "rs7155418",
        "rs73099715",
        "rs773588",
        "rs55693740",
        "rs71624119",
        "rs11678791",
        "rs11711534",
        "rs2793108",
        "rs35673421",
        "rs2393967",
        "rs7556897",
        "rs6025",
        "rs73082914",
        "rs12198173",
        "rs10025152",
        "rs9262132",
        "rs118154869",
        "rs1297258",
        "rs6748088",
        "rs11087101",
        "rs2041733",
        "rs12720356",
        "rs61907765",
        "rs6818271",
        "rs9611137",
        "rs1980615",
        "rs80311338",
        "rs4847240",
        "rs2024092",
        "rs454715",
        "rs11843351",
        "rs117675359",
        "rs2297775",
        "rs3117073",
        "rs1250573",
        "rs75245322",
        "rs1547050",
        "rs2302712",
        "rs4703855",
        "rs2005535",
        "rs6556417",
        "rs12206238",
        "rs12724079",
        "rs17517792",
        "rs72793807",
        "rs4755450",
        "rs10010325",
        "rs1264372",
        "rs9557201",
        "rs1982995",
        "rs12551330",
        "rs11743296",
        "rs1540293",
        "rs6441973",
        "rs9814873",
        "rs829418",
        "rs36123466",
        "rs111602786",
        "rs60319976",
        "rs1042157",
        "rs17057051",
        "rs7292732",
        "rs2296330",
        "rs151407",
        "rs181826",
        "rs17391694",
        "rs7785730",
        "rs1946410",
        "rs8055876",
        "rs61894547",
        "rs7699742",
        "rs17712103",
        "rs720201",
        "rs17731449",
        "rs11549673",
        "rs10190751",
        "rs10479001",
        "rs11970194",
        "rs3129791",
        "rs7240004",
        "rs77813979",
        "rs62434177",
        "rs72661359",
        "rs12149258",
        "rs730086",
        "rs6074022",
        "rs4401177",
        "rs7194167",
        "rs7580636",
        "rs10908465",
        "rs2759665",
        "rs36118932",
        "rs12599586",
        "rs442694",
        "rs516246",
        "rs118152628",
        "rs16940935",
        "rs10054168",
        "rs7703043",
        "rs1504215",
        "rs3900628",
        "rs778151",
        "rs9503530",
        "rs438475",
        "rs9264942",
        "rs2267364",
        "rs4781128",
        "rs10416073",
        "rs9350354",
        "rs1059612",
        "rs2031797",
        "rs348588",
        "rs8127691",
        "rs780094",
        "rs12948602",
        "rs4949658",
        "rs72793687",
        "rs75691080",
        "rs75119918",
        "rs2395022",
        "rs6785874",
        "rs849135",
        "rs11117431",
        "rs663310",
        "rs13008738",
        "rs12103",
        "rs3124994",
        "rs13087930",
        "rs6682968",
        "rs430092",
        "rs2854050",
        "rs303429",
        "rs11130254",
        "rs1847472",
        "rs12039893",
        "rs10051722",
        "rs9465925",
        "rs1292034",
        "rs35507033",
        "rs9276427",
        "rs11716837",
        "rs1187411",
        "rs12439281",
        "rs9348876",
        "rs1573646",
        "rs4443541",
        "rs59867199",
        "rs1646019",
        "rs3131618",
        "rs10185424",
        "rs74797399",
        "rs4795397",
        "rs532098",
        "rs1666559",
        "rs630044",
        "rs13213152",
        "rs9262582",
        "rs7711427",
        "rs2946365",
        "rs998235",
        "rs77981966",
        "rs2143606",
        "rs72807491",
        "rs9261947",
        "rs10210680",
        "rs13001325",
        "rs3197999",
        "rs10758669",
        "rs538649",
        "rs11560908",
        "rs1548914",
        "rs59979159",
        "rs62142685",
        "rs2392896",
        "rs490608",
        "rs3739706",
        "rs78040933",
        "rs2836757",
        "rs66901253",
        "rs2395228",
        "rs180515",
        "rs10743181",
        "rs72885270",
        "rs463543",
        "rs6738825",
        "rs7758080",
        "rs9262143",
        "rs587412",
        "rs10143836",
        "rs78686200",
        "rs244080",
        "rs78102288",
        "rs4409689",
        "rs913678",
        "rs1813006",
        "rs10774625",
        "rs11715725",
        "rs6846971",
        "rs2538470",
        "rs11601686",
        "rs79679335",
        "rs118052167",
        "rs7933433",
        "rs4663698",
        "rs10492682",
        "rs6476844",
        "rs17572109",
        "rs1322063",
        "rs4802307",
        "rs72650663",
        "rs72807046",
        "rs34850016",
        "rs210837",
        "rs34779708",
        "rs2239842",
        "rs7905",
        "rs7434107",
        "rs7977247",
        "rs62578666",
        "rs118070864",
        "rs71324929",
        "rs4672440",
        "rs6691768",
        "rs35848181",
        "rs1010473",
        "rs2243198",
        "rs3801944",
        "rs570963",
        "rs7515488",
        "rs36016881",
        "rs117917460",
        "rs13204742",
        "rs56243424",
        "rs8047363",
        "rs16868943",
        "rs2497318",
        "rs194749",
        "rs17081657",
        "rs7194886",
        "rs117224507",
        "rs445150",
        "rs6904596",
        "rs1842076",
        "rs59682551",
        "rs6534338",
        "rs17674015",
        "rs11611855",
        "rs34592089",
        "rs7760902",
        "rs4821544",
        "rs13034020",
        "rs2179070",
        "rs6702421",
        "rs12967204",
        "rs62323898",
        "rs6838613",
        "rs587259",
        "rs3742130",
        "rs636918",
        "rs640466",
        "rs80575",
        "rs12134279",
        "rs11691685",
        "rs12827878",
        "rs116630553",
        "rs16922581",
        "rs6913635",
        "rs11679753",
        "rs3922",
        "rs72687906",
        "rs17185574",
        "rs757256",
        "rs2755244",
        "rs12924003",
        "rs6595951",
        "rs4599384",
        "rs76521383",
        "rs1059231",
        "rs2296748",
        "rs8051363",
        "rs1567009",
        "rs10494164",
        "rs2193046",
        "rs73072475",
        "rs11719729",
        "rs566416",
        "rs72727394",
        "rs4934167",
        "rs694739",
        "rs9869826",
        "rs2523619",
        "rs6062496",
        "rs9607637",
        "rs10465507",
        "rs72727387",
        "rs2058813",
        "rs78251404",
        "rs3132610",
        "rs2847278",
        "rs17187115",
        "rs6866614",
        "rs28399256",
        "rs4120852",
        "rs11952467",
        "rs11548392",
        "rs11804831",
        "rs115992102",
        "rs4713813",
        "rs6908425",
        "rs773424",
        "rs67643815",
        "rs13407913",
        "rs10053519",
        "rs6693292",
        "rs2227551",
        "rs3181356",
        "rs10412803",
        "rs10484399",
        "rs11876243",
        "rs114297139",
        "rs1572343",
        "rs17622378",
        "rs6688636",
        "rs2027499",
        "rs5743291",
        "rs2236379",
        "rs4239225",
        "rs78439216",
        "rs7555634",
        "rs314313",
        "rs969577",
        "rs102275",
        "rs1633036",
        "rs2071538",
        "rs941823",
        "rs12120143",
        "rs10908471",
        "rs17696736",
        "rs4851597",
        "rs7438704",
        "rs12214723",
        "rs12137256",
        "rs2581774",
        "rs3095273",
        "rs113688608",
        "rs11800642",
        "rs2395030",
        "rs1070444",
        "rs77535416",
        "rs7794532",
        "rs465474",
        "rs75889859",
        "rs34725546",
        "rs10169608",
        "rs4676410",
        "rs3761705",
        "rs10877013",
        "rs459048",
        "rs3751092",
        "rs266433",
        "rs13109404",
        "rs919115",
        "rs3732379",
        "rs7275685",
        "rs140143",
        "rs7552536",
        "rs9257136",
        "rs2068834",
        "rs13028000",
        "rs3130117",
        "rs653178",
        "rs2055647",
        "rs34619612",
        "rs2029923",
        "rs7268588",
        "rs3131820",
        "rs26528",
        "rs9889296",
        "rs3129716",
        "rs9368699",
        "rs149943",
        "rs72968949",
        "rs10912561",
        "rs1150257",
        "rs6470634",
        "rs7253253",
        "rs3794436",
        "rs11633987",
        "rs7252175",
        "rs2413583",
        "rs2230683",
        "rs2848713",
        "rs12766391",
        "rs80089053",
        "rs16948451",
        "rs7993214",
        "rs835573",
        "rs281392",
        "rs13126505",
        "rs11145765",
        "rs12208739",
        "rs1569328",
        "rs7517847",
        "rs2850950",
        "rs12151689",
        "rs6893779",
        "rs172933",
        "rs7744001",
        "rs17491356",
        "rs3853824",
        "rs722788",
        "rs9391734",
        "rs949635",
        "rs2084318",
        "rs117762340",
        "rs9491892",
        "rs12418638",
        "rs233807",
        "rs116046827",
        "rs56103919",
        "rs3784921",
        "rs12150495",
        "rs7802715",
        "rs33942096",
        "rs11185982",
        "rs270654",
        "rs7617915",
        "rs2974935",
        "rs7349418",
        "rs13262484",
        "rs7773324",
        "rs476236",
        "rs4984246",
        "rs72834743",
        "rs75396339",
        "rs1363907",
        "rs3094035",
        "rs57844307",
        "rs6656890",
        "rs13086240",
        "rs61802091",
        "rs3118357",
        "rs117257494",
        "rs12585310",
        "rs57794755",
        "rs12139286",
        "rs34776628",
        "rs2641348",
        "rs421446",
        "rs2571377",
        "rs395157",
        "rs11229555",
        "rs71421920",
        "rs6827756",
        "rs25887",
        "rs1517037",
        "rs76906269",
        "rs13197574",
        "rs11209009",
        "rs3847953",
        "rs10899219",
        "rs12893301",
        "rs114793712",
        "rs2026029",
        "rs9289629",
        "rs62142686",
        "rs745155",
        "rs9263455",
        "rs2027856",
        "rs12118913",
        "rs2769267",
        "rs17443185",
        "rs3776414",
        "rs9320141",
        "rs10489912",
        "rs34787213",
        "rs17792389",
        "rs12718244",
        "rs76181804",
        "rs12272129",
        "rs12947480",
        "rs34856868",
        "rs13211507",
        "rs1267501",
        "rs2516698",
        "rs3094024",
        "rs8083571",
        "rs714027",
        "rs915286",
        "rs34699226",
        "rs3135309",
        "rs12237953",
        "rs6707803",
        "rs7833266",
        "rs3129788",
        "rs72691846",
        "rs75932609",
        "rs974801",
        "rs74356516",
        "rs78122011",
        "rs9554587",
        "rs9673419",
        "rs5757650",
        "rs6709988",
        "rs7195296",
        "rs34089926",
        "rs564349",
        "rs7749305",
        "rs78135710",
        "rs72978783",
        "rs1769016",
        "rs6740462",
        "rs113593463",
        "rs3130250",
        "rs10985112",
        "rs8109501",
        "rs7608910",
        "rs7848647",
        "rs8055982",
        "rs442745",
        "rs1799964",
        "rs3116830",
        "rs12624433",
        "rs868150",
        "rs568617",
        "rs6656966",
        "rs2298952",
        "rs67572105",
        "rs10057047",
        "rs2638324",
        "rs4243971",
        "rs10758661",
        "rs56167332",
        "rs11748553",
        "rs9366394",
        "rs1128905",
        "rs6738721",
        "rs6915823",
        "rs13472",
        "rs10412574",
        "rs13199524",
        "rs6760732",
        "rs7170683",
        "rs1292053",
        "rs11656146",
        "rs2289093",
        "rs3790609",
        "rs12138260",
        "rs16946807",
        "rs76174160",
        "rs117914011",
        "rs3915617",
        "rs11971702",
        "rs4767956",
        "rs10807124",
        "rs7106446",
        "rs224053",
        "rs34725611",
        "rs6705001",
        "rs41294605",
        "rs12040750",
        "rs10748096",
        "rs75378030",
        "rs1892337",
        "rs1531270",
        "rs11768997",
        "rs72795177",
        "rs3806110",
        "rs13204048",
        "rs12042319",
        "rs28999107",
        "rs2300605",
        "rs79000843",
        "rs3792782",
        "rs13175903",
        "rs17630235",
        "rs7186889",
        "rs1157509",
        "rs11174631",
        "rs401775",
        "rs7805803",
        "rs10276381",
        "rs13105682",
        "rs73101534",
        "rs67707912",
        "rs76458677",
        "rs13397985",
        "rs17103120",
        "rs12949918",
        "rs3864261",
        "rs76922682",
        "rs2784114",
        "rs12239114",
        "rs74751235",
        "rs16988402",
        "rs6679677",
        "rs67948565",
        "rs6657670",
        "rs11066320",
        "rs79044169",
        "rs252214",
        "rs11536857",
        "rs2280246",
        "rs11264305",
        "rs2726032",
        "rs6058869",
        "rs3790514",
        "rs2270366",
        "rs5005770",
        "rs204996",
        "rs13113094",
        "rs1528602",
        "rs1032070",
        "rs11597065",
        "rs1919127",
        "rs17040797",
        "rs78075599",
        "rs2779253",
        "rs1610677",
        "rs1865223",
        "rs67183459",
        "rs374326",
        "rs2221134",
        "rs880051",
        "rs4643314",
        "rs1465701",
        "rs767019",
        "rs114089912",
        "rs11159833",
        "rs11961168",
        "rs10835755",
        "rs6856616",
        "rs3131814",
        "rs11231713",
        "rs2073505",
        "rs7633243",
        "rs3131854",
        "rs9833750",
        "rs2625276",
        "rs118128985",
        "rs633372",
        "rs6457590",
        "rs4976646",
        "rs35074907",
        "rs13219354",
        "rs11264426",
        "rs4917129",
        "rs116780602",
        "rs1362104",
        "rs11681263",
        "rs367254",
        "rs62427027",
        "rs11064881",
        "rs6561151",
        "rs918490",
        "rs12796489",
        "rs10883221",
        "rs27659",
        "rs13215208",
        "rs2274351",
        "rs6500315",
        "rs11788118",
        "rs1838978",
        "rs6584303",
        "rs35283189",
        "rs608337",
        "rs212388",
        "rs11065979",
        "rs1736911",
        "rs2894083",
        "rs116482870",
        "rs3805226",
        "rs6737414",
        "rs2633958",
        "rs75208336",
        "rs2881698",
        "rs2005557",
        "rs6682330",
        "rs12918327",
        "rs372889",
        "rs17689550",
        "rs3184504",
        "rs35164067",
        "rs73108339",
        "rs142328479",
        "rs17676923",
        "rs7204722",
        "rs73082941",
        "rs10147645",
        "rs2248465",
        "rs3811406",
        "rs2228145",
        "rs11869582",
        "rs61959448",
        "rs6111031",
        "rs200990",
        "rs3130888",
        "rs72799938",
        "rs7253596",
        "rs2082881",
        "rs559928",
        "rs3132392",
        "rs7914814",
        "rs1158199",
        "rs78539283",
        "rs61815610",
        "rs12206077",
        "rs10762563",
        "rs2945412",
        "rs3181354",
        "rs925255",
        "rs1624116",
        "rs354033",
        "rs28373355",
        "rs111340452",
        "rs729023",
        "rs11689575",
        "rs34299154",
        "rs4327730",
        "rs17205284",
        "rs11589479",
        "rs4781011",
        "rs12525504",
        "rs2384288",
        "rs17032011",
        "rs11236797",
        "rs9408254",
        "rs12985909",
        "rs8052975",
        "rs7753008",
        "rs7322781",
        "rs10985085",
        "rs4957300",
        "rs73079076",
        "rs803137",
        "rs6091836",
        "rs1509635",
        "rs10908481",
        "rs631106",
        "rs2739522",
        "rs267219",
        "rs72975936",
        "rs7969592",
        "rs77397067",
        "rs4423337",
        "rs732072",
        "rs921720",
        "rs10800309",
        "rs259964",
        "rs2961703",
        "rs10089519",
        "rs9262141",
        "rs251339",
        "rs6512265",
        "rs41295125",
        "rs2153283",
        "rs3785794",
        "rs2476601",
        "rs9267677",
        "rs10208137",
        "rs2891409",
        "rs7727678",
        "rs9379856",
        "rs11959014",
        "rs6031301",
        "rs9494844",
        "rs4806703",
        "rs4664304",
        "rs8070345",
        "rs10748781",
        "rs35320439",
        "rs35320232",
        "rs17748089",
        "rs17090837",
        "rs3095330",
        "rs13013484",
        "rs45515895",
        "rs2071556",
        "rs3801810",
        "rs4719879",
        "rs1262475",
        "rs2836878",
        "rs56090761",
        "rs188245",
        "rs10094579",
        "rs9457247",
        "rs17422797",
        "rs62076937",
        "rs33965856",
        "rs295320",
        "rs968567",
        "rs7101544",
        "rs62150971",
        "rs3124996",
        "rs9907966",
        "rs71326907",
        "rs1235162",
        "rs2230365",
        "rs1233330",
        "rs5024432",
        "rs3749971",
        "rs77648435",
        "rs1016216",
        "rs2404233",
        "rs185789577",
        "rs11120029",
        "rs11152949",
        "rs17663669",
        "rs9567245",
        "rs7036761",
        "rs12194548",
        "rs34013267",
        "rs1619376",
        "rs6456426",
        "rs10798069",
        "rs1517352",
        "rs56764892",
        "rs77445405",
        "rs12127370",
        "rs11713774",
        "rs3130773",
        "rs2227284",
        "rs7158822",
        "rs116090320",
        "rs3094204",
        "rs7547569",
        "rs3818177",
        "rs3117143",
        "rs117434348",
        "rs116625780",
        "rs887783",
        "rs36132266",
        "rs45559341",
        "rs13194504",
        "rs34241101",
        "rs4894786",
        "rs903506",
        "rs1879145",
        "rs7948288",
        "rs6697145",
        "rs11587213",
        "rs7134408",
        "rs4692386",
        "rs4845604",
        "rs12199775",
        "rs7165170",
        "rs233722",
        "rs60969203",
        "rs267949",
        "rs13263338",
        "rs116736594",
        "rs73745198",
        "rs3130893",
        "rs112226504",
        "rs2886554",
        "rs6916345",
        "rs7752195",
        "rs7853287",
        "rs17730134",
        "rs11154761",
        "rs17693963",
    ]
)


def generate_c_and_t_prs_scores(
    assembly: str,
    trait: str,
    pmid: str,
    ancestry: AncestryResults,
    dosage_matrix_path: str,
    disease_prevalence: float,
    continuous_trait: bool,
    index_name: str,
    training_populations: list[str] | None = None,
    distance_based_cluster: bool = True,
    ld_window_bp: int = 20_000,
    experiment_mapping: ExperimentMapping | None = None,
    min_abs_beta: float = 0.01,
    max_abs_beta: float = 3.0,
    p_value_threshold: float = 0.05,
    sample_chunk_size: int = 200,
    user: CachedAuth | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
    reporter: ProgressReporter | None = None,
    debug: bool = True,
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
    # This part goes through dosage matrix the first time to get overlapping loci
    with Timer() as timer:
        gwas_scores_path = get_sumstats_file(trait, assembly, pmid)
        scores = _load_association_scores(str(gwas_scores_path))
    logger.debug("Time to load association scores: %s", timer.elapsed_time)

    if reporter is not None:
        reporter.message.remote("Loaded association scores")  # type: ignore

    with Timer() as timer:
        # prune preprocessed_scores to only include loci with p-values below the threshold
        # and filter down to sites with beta values within range
        scores = scores[scores[P_COLUMN] <= p_value_threshold]
        scores = scores[
            (scores[BETA_COLUMN].abs() >= min_abs_beta) & (scores[BETA_COLUMN].abs() <= max_abs_beta)
        ]
        if distance_based_cluster:
            # if debugging against Dave Cutler's PRS calculator, uncomment to keep only those snps that
            # are in the cutler_snp_set, based on the VARIANT_ID_COLUMN
            # preprocessed_scores = preprocessed_scores[preprocessed_scores[VARIANT_ID_COLUMN].isin(_cutler_snp_set)] # noqa
            scores = prune_by_window(scores, ld_window_bp)

            # TODO: check for non-direct match and warn
            preprocessed_scores = _preprocess_scores(scores)
        else:
            if training_populations is None:
                raise ValueError(
                    "If distance_based_cluster is False, training_populations must be provided"
                )

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
                raise ValueError(
                    f"{sumstat_population} is likely not supported. Failed to get map file: {e}"
                )

            if index_name is not None and cluster_opensearch_config is None and user is None:
                raise ValueError(
                    "If index_name is provided, either user or cluster_opensearch_config must be provided."
                )

            # TODO: check for non-direct match and warn
            preprocessed_scores = _preprocess_scores(scores)

            scores_after_c_t, _loci_and_allele_comparison = ld_clump(
                preprocessed_scores, str(sumstat_ld_map_path)
            )

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

    if debug:
        preprocessed_scores.to_csv("preprocessed_scores.csv")
        population_allele_frequencies.to_csv("population_allele.tsv", sep="\t")
        ancestry_weighted_afs.to_csv("ancestry_weighted_afs.tsv", sep="\t")
        mean_q.to_csv("mean_q.tsv", sep="\t")
        preprocessed_scores.to_csv("preprocessed_scores_step4.csv")
        Va.to_csv("Va.tsv", sep="\t")

        print("Ve", Ve)
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

    if reporter is not None:
        reporter.message.remote("Fetched allele frequencies")  # type: ignore

    # Accumulate the results
    prs_scores: dict[str, float] = {}
    corrected_odds_ratios: dict[str, float] = {}

    score_loci_filter = pc.field(GENOTYPE_DOSAGE_LOCUS_COLUMN).isin(
        pa.array(list(dosage_loci_nonmissing_afs))
    )
    dosage_ds = ds.dataset(dosage_matrix_path, format="feather").filter(score_loci_filter)
    with Timer() as outer_timer:
        samples_processed = 0

        for sample_pop_group in sample_groups:
            population, sample_group = sample_pop_group

            with Timer() as timer:
                sample_genotypes = dosage_ds.to_table(
                    [GENOTYPE_DOSAGE_LOCUS_COLUMN, *sample_group]
                ).to_pandas()
                sample_genotypes = sample_genotypes.rename(
                    columns={GENOTYPE_DOSAGE_LOCUS_COLUMN: SNPID_COLUMN}
                )
                sample_genotypes = sample_genotypes.set_index(SNPID_COLUMN)

                # TODO @akotlar: 2024-08-06 impute genotypes
                sample_genotypes = sample_genotypes.replace(-1, np.nan)

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
                adjusted_genotypes = (
                    sample_genotypes
                    - 2.0 * ancestry_weighted_afs.loc[sample_genotypes.index, sample_genotypes.columns]
                )
                beta_values = preprocessed_scores.loc[adjusted_genotypes.index][BETA_COLUMN]

                # prs_score for individual samples
                # beta_values is a vector of beta values for each locus n * 1
                # adjusted_genotypes is a matrix of genotypes for each sample n * m
                # Multiply genotype DataFrame by SNP values DataFrame
                prs_scores_chunk = adjusted_genotypes.multiply(beta_values, axis=0)

                # Replace NaN with 0
                prs_scores_chunk = prs_scores_chunk.fillna(0)

                # Sum the results across the SNPs for each sample
                prs_scores_chunk = prs_scores_chunk.sum(axis=0)

                sample_prevalence = 1.0 - norm.cdf(threshold, prs_scores_chunk, Ve)

                real_or = or_prev * sample_prevalence / (1.0 - sample_prevalence)

                # add each prs score to the prs_scores dict
                for index, sample in enumerate(adjusted_genotypes.columns):
                    prs_scores[sample] = prs_scores_chunk.loc[sample]
                    corrected_odds_ratios[sample] = real_or[index]

                if debug:
                    sample_genotypes.to_csv(f"sample_genotypes_{samples_processed}.tsv", sep="\t")
                    adjusted_genotypes.to_csv(f"adjusted_genotypes{samples_processed}.tsv", sep="\t")
                    beta_values.to_csv(f"beta_values_{samples_processed}.tsv", sep="\t")
                    prs_scores_chunk.to_csv(f"prs_scores_chunk_{samples_processed}.tsv", sep="\t")
                    print("ancestry_weighted_afs", ancestry_weighted_afs)
                    print("adjusted_genotypes", adjusted_genotypes)
                    print(
                        "For chunk",
                        samples_processed,
                        "sample_prevalence",
                        sample_prevalence,
                        "threshold",
                        threshold,
                        "prs_scores_chunk",
                        prs_scores_chunk,
                        "Ve",
                        Ve,
                    )
                    preprocessed_scores.loc[adjusted_genotypes.index].to_csv(
                        f"preprocessed_scores_{samples_processed}.tsv", sep="\t"
                    )
                    print("prs_scores_chunk", prs_scores_chunk)
                    print("sample_prevalence", sample_prevalence)
                    print("real_or", real_or)
                    print("corrected_odds_ratios", corrected_odds_ratios)

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
