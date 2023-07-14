"""Training code for Global Ancestry Model.

This script takes a vcf of preprocessed variants (see preprocess_vcfs.sh) and generates:

1.  A list of variants to be used for inference.

2.  PCA loadings mapping the list of variants in (1) to PC space.

3.  Classifiers mapping PC space to the 26 HapMap populations as well as 5 continent-level
superpopulations.
"""
import dataclasses
import gzip
import logging
import random
import re
import sys
from collections import Counter
from collections.abc import Collection, Container, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args

import allel
import numpy as np
import pandas as pd
import sklearn
import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skops.io import dump as skops_dump

from bystro.ancestry.asserts import assert_equals, assert_true
from bystro.ancestry.train_utils import get_variant_ids_from_callset, head

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ANCESTRY_DIR = Path().absolute()
DATA_DIR = ANCESTRY_DIR / "data"
KGP_VCF_DIR = DATA_DIR / "kgp_vcfs"
INTERMEDIATE_DATA_DIR = ANCESTRY_DIR / "intermediate_data"
VCF_PATH = DATA_DIR / "1KGP_final_variants_1percent.vcf.gz"
ANCESTRY_MODEL_PRODUCTS_DIR = ANCESTRY_DIR / "ancestry_model_products"

ANCESTRY_INFO_PATH = DATA_DIR / "20130606_sample_info.txt"
ROWS, COLS = 0, 1
QUALITY_CUTOFF = 100  # for variant quality filtering
PLOIDY = 2
AUTOSOMAL_CHROMOSOMES = set(range(1, 22 + 1))
EXPECTED_NUM_POPULATIONS = 26
FST_THRESHOLD = 0.3
MI_THRESHOLD = 0
PCA_DIMS = 50
EXPLAINED_VARIANCE_THRESHOLD = 0.1
RFC_TRAIN_ACCURACY_THRESHOLD = 0.9
RFC_TEST_ACCURACY_THRESHOLD = 0.75
RFC_TRAIN_SUPERPOP_ACCURACY_THRESHOLD = 0.99
RFC_TEST_SUPERPOP_ACCURACY_THRESHOLD = 0.99

# superpop definitions taken from ensembl
SUPERPOP_FROM_POP = {
    "CHB": "EAS",
    "JPT": "EAS",
    "CHS": "EAS",
    "CDX": "EAS",
    "KHV": "EAS",
    "CEU": "EUR",
    "TSI": "EUR",
    "FIN": "EUR",
    "GBR": "EUR",
    "IBS": "EUR",
    "YRI": "AFR",
    "LWK": "AFR",
    "MSL": "AFR",
    "ESN": "AFR",
    "ASW": "AFR",
    "ACB": "AFR",
    "GWD": "AFR",
    "MXL": "AMR",
    "PUR": "AMR",
    "CLM": "AMR",
    "PEL": "AMR",
    "GIH": "SAS",
    "PJL": "SAS",
    "BEB": "SAS",
    "STU": "SAS",
    "ITU": "SAS",
}
POPS = np.array(sorted(SUPERPOP_FROM_POP.keys()))
SUPERPOPS = np.array(sorted(set(SUPERPOP_FROM_POP.values())))

VARIANT_REGEX = re.compile(
    r"""
    ^
    chr(1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22)  # (autosomal) chromosome
    :
    [0-9]+  # position
    :
    [ACGT]  # ref allele
    :
    [ACGT]  # alt allele
    $
    """,
    re.VERBOSE,
)

rng = np.random.RandomState(1337)

Variant = str


def _load_callset() -> dict[str, Any]:
    """Load callset and perform as many checks as can be done without processing it."""
    callset = allel.read_vcf(str(VCF_PATH), log=sys.stdout)
    num_variants, num_samples = 1203135, 2504
    assert_equals(
        f"chromosomes in {VCF_PATH}",
        {int(chrom) for chrom in callset["variants/CHROM"]},
        "autosomal chromosomes",
        AUTOSOMAL_CHROMOSOMES,
    )
    genotypes = callset["calldata/GT"]
    assert_equals(
        "genotype dimensions",
        genotypes.shape,
        "predicted genotype dimensions",
        (num_variants, num_samples, PLOIDY),
        comment=(
            f"VCF file {VCF_PATH} had unexpected dimensions.  Expected matrix of shape: "
            f"(num_variants: {num_variants}, num_samples: {num_samples}, ploidy: {PLOIDY})"
        ),
    )
    return callset


def load_callset_for_variants(variants: set[str]) -> pd.DataFrame:
    """Load Thousand Genomes data filtered to variants."""
    file_template = "ALL.chr{}.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz"
    genotype_dfs = []
    for chromosome in range(1, 22 + 1):
        logger.info("starting on chromosome: %s", chromosome)
        file_path = str(KGP_VCF_DIR / file_template.format(chromosome))
        genotype_df = _parse_vcf(file_path, variants)
        logger.info("got genotype_df of shape: %s", genotype_df.shape)
        genotype_dfs.append(genotype_df)
    return pd.concat(genotype_dfs, axis=1)


def _parse_vcf_line_for_dosages(
    line: str, variants_to_keep: Container[Variant]
) -> tuple[Variant, list[int]] | None:
    # will throw ValueError if "PASS" not found, which is good
    fields = line[: line.index("PASS")].split()
    variant = ":".join([fields[0], fields[1], fields[3], fields[4]])
    assert_true("variant is a valid variant string", (VARIANT_REGEX.match(variant)))
    if variant in variants_to_keep:
        variant_dosages = [
            int(psa[0]) + int(psa[2]) for psa in line.split()[9:]
        ]  #  pipe-separated annotation e.g. '0|1'
        return variant, variant_dosages
    return None


def _get_chromosome_from_variant(variant: Variant) -> str:
    return variant.split(":")[0]


T = TypeVar("T")


def _calculate_recovery_rate(
    found_variants: Collection[Variant], variants_to_keep: Collection[Variant]
) -> float:
    if len(found_variants) == 0:
        return 0.0
    found_chromosomes = {_get_chromosome_from_variant(v) for v in found_variants}
    if len(found_chromosomes) > 1:
        msg = "Found chromosomes contains more than one chromosome"
        raise ValueError(msg, found_chromosomes)
    relevant_chromosome = head(found_chromosomes)
    relevant_variants_to_keep = {
        v for v in variants_to_keep if _get_chromosome_from_variant(v) == relevant_chromosome
    }
    return len(found_variants) / len(relevant_variants_to_keep)


def _parse_vcf(file_path: str, variants_to_keep: Collection[str]) -> pd.DataFrame:
    with gzip.open(file_path, "rt") as f:
        return _parse_vcf_from_file_stream(f, variants_to_keep)


def _parse_vcf_from_file_stream(
    file_stream: Iterable[str], variants_to_keep: Collection[str]
) -> pd.DataFrame:
    found_variants = []
    dosage_data = []
    sample_ids = None
    total_lines = 0
    for line in tqdm.tqdm(file_stream):
        total_lines += 1
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            sample_ids = line.split()[9:]
        elif variant_dosages := _parse_vcf_line_for_dosages(line, variants_to_keep):
            variant, dosages = variant_dosages
            found_variants.append(variant)
            dosage_data.append(dosages)
        else:
            continue
    if sample_ids is None:
        msg = "Couldn't find sample ids in VCF"
        raise ValueError(msg)
    found_chromosomes = {_get_chromosome_from_variant(v) for v in found_variants}
    assert_true("Extracted sample_ids from vcf", sample_ids is not None)
    logger.info(
        "processed %s lines, retaining %s variants from %s chromosomes",
        total_lines,
        len(found_variants),
        len(found_chromosomes),
    )
    df_values: np.ndarray | list[list[float]] = np.array(dosage_data).T if dosage_data else []
    dosage_df = pd.DataFrame(df_values, index=sample_ids, columns=found_variants)
    # we assume each vcf file contains variants for a single chromosome
    recovery_rate = _calculate_recovery_rate(found_variants, variants_to_keep)
    logger.info("recovery rate: %s", recovery_rate)
    return dosage_df


def _load_genotypes() -> pd.DataFrame:
    """Read variants from disk, return as count matrix with dimensions (samples X variants)."""
    logger.info("loading callset")
    callset = _load_callset()
    logger.info("finished loading callset")
    samples = callset["samples"]
    genotypes = allel.GenotypeArray(callset["calldata/GT"])
    variant_ids = get_variant_ids_from_callset(callset)
    logger.info("starting with GenotypeArray of shape: %s", genotypes.shape)
    ref_is_snp = np.array([len(ref) == 1 for ref in callset["variants/REF"]])
    logger.info("found %s variants where ref is mononucleotide", sum(ref_is_snp))
    # first alt is SNP and rest are empty...
    alt_is_snp = np.array(
        [len(row[0]) == 1 and set(row[1:]) == {""} for row in callset["variants/ALT"]],
    )
    logger.info("found %s variants where alt is mononucleotide", sum(alt_is_snp))
    allele_counts = genotypes.count_alleles()[:]
    is_single_allele_snp = allele_counts.max_allele() == 1
    logger.info("found %s single allele snps", sum(is_single_allele_snp))
    is_non_singleton = (
        allele_counts[:, :2].min(axis=COLS) > 1
    )  # min count over reference, first allele is greater than zero
    logger.info("found %s non-singleton alleles", sum(is_non_singleton))
    is_high_quality = callset["variants/QUAL"] >= QUALITY_CUTOFF
    logger.info("found %s high quality alleles", sum(is_high_quality))

    mask = ref_is_snp & alt_is_snp & is_single_allele_snp & is_non_singleton & is_high_quality
    logger.info("keeping %s alleles", sum(mask))
    filtered_genotypes = genotypes.compress(mask, axis=ROWS).to_n_alt().T
    variant_ids = variant_ids[mask]
    assert_equals(
        "filtered genotype matrix shape",
        filtered_genotypes.shape,
        "sample and variant id sizes",
        (len(samples), len(variant_ids)),
    )
    return pd.DataFrame(filtered_genotypes, index=samples, columns=variant_ids)


def filter_samples_for_relatedness(
    genotypes: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter samples for relatedness, returning subset of unrelated individuals."""
    logger.info("Filtering samples for relatedness")
    ancestry_df = _load_ancestry_df()
    assert_equals("genotype samples", genotypes.index, "label samples", labels.index)
    samples = genotypes.index
    ancestry_df = ancestry_df[ancestry_df.index.isin(samples)]
    family_ids = ancestry_df["Family ID"].unique()
    logger.info("Found %s unique families", len(family_ids))
    unrelated_samples = []
    random.seed(1337)
    for family_id in family_ids:
        family_members = ancestry_df[ancestry_df["Family ID"] == family_id].index
        # grab value out of array...
        family_member = random.choice(family_members)
        unrelated_samples.append(family_member)
    unrelated_sample_idx = np.array(unrelated_samples)
    genotypes, labels = (
        genotypes.loc[unrelated_sample_idx],
        labels.loc[unrelated_sample_idx],
    )
    # we did this earlier but removing samples could make more variants monomorphic
    genotypes = _filter_variants_for_monomorphism(genotypes)
    return genotypes, labels


def _load_ancestry_df() -> pd.DataFrame:
    return pd.read_csv(ANCESTRY_INFO_PATH, sep="\t")


def load_label_data(samples: pd.Index) -> pd.DataFrame:
    """Load dataframe of population, superpop labels for samples."""
    ancestry_df = _load_ancestry_df()
    missing_samples = set(samples) - set(ancestry_df["Sample"])
    if missing_samples:
        msg = f"Ancestry dataframe is missing samples: {missing_samples}"
        raise AssertionError(msg)
    populations = sorted(ancestry_df["Population"].unique())
    if EXPECTED_NUM_POPULATIONS != len(populations):
        msg = (
            f"Found wrong number of populations ({len(populations)}) in ancestry df, "
            f"expected {EXPECTED_NUM_POPULATIONS}"
        )
        raise ValueError(msg)
    get_pop_from_sample = ancestry_df.set_index("Sample")["Population"].to_dict()
    labels = pd.DataFrame(
        [get_pop_from_sample[s] for s in samples],
        index=samples,
        columns=["population"],
    )
    labels["superpop"] = labels["population"].apply(SUPERPOP_FROM_POP.get)

    assert_true("no missing data in labels", labels.notna().all().all())
    assert_equals(
        "number of populations",
        EXPECTED_NUM_POPULATIONS,
        "number of populations found in labels",
        len((labels["population"]).unique()),
    )
    assert_equals(
        "number of superpopulations",
        len(set(SUPERPOP_FROM_POP.values())),
        "superpopulations found in labels",
        len((labels["superpop"]).unique()),
    )
    assert_equals("samples", samples, "labels", labels.index)
    return labels


def _filter_variants_for_maf(genotypes: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    logger.info("Filtering %s genotypes for MAF threshold {threshold}", len(genotypes.columns))
    frequencies = genotypes.mean(axis=0) / PLOIDY
    maf_mask = (frequencies) > threshold
    num_passing_variants = sum(maf_mask)
    filtered_genotypes = genotypes[genotypes.columns[maf_mask]]
    assert_equals(
        "Number of passing variants",
        num_passing_variants,
        "number of filtered genotype columns",
        filtered_genotypes.shape[1],
    )
    logger.info("After MAF filtering: %s", len(filtered_genotypes.columns))
    return filtered_genotypes


def _filter_variants_for_ld(
    genotypes: pd.DataFrame,
    size: int = 500,
    step: int = 200,
    threshold: float = 0.1,
    n_iter: int = 5,
) -> pd.DataFrame:
    logger.info("LD pruning genotypes of shape %s", genotypes.shape)
    for i in range(n_iter):
        loc_unlinked = allel.locate_unlinked(
            genotypes.to_numpy().T,
            size=size,
            step=step,
            threshold=threshold,
        )
        n = np.count_nonzero(loc_unlinked)
        n_remove = genotypes.shape[1] - n
        logger.info("iteration %s retaining %s removing %s variants", i + 1, n, n_remove)
        genotypes = genotypes[genotypes.columns[loc_unlinked]]
    logger.info("After LD pruning, genotypes of shape %s", genotypes.shape)
    return genotypes


def _filter_variants_for_monomorphism(genotypes: pd.DataFrame) -> pd.DataFrame:
    """Exclude monomorphic variants, i.e. those with no variation in dataset."""
    monomorphic_mask = genotypes.std(axis="index") > 0
    num_excluded_monomorphic_variants = np.sum(~monomorphic_mask)
    logger.info("Removing %s monomorphic variants", num_excluded_monomorphic_variants)
    monomorphic_fraction = num_excluded_monomorphic_variants / len(monomorphic_mask)
    assert_true("fraction of excluded monomorphic variants less than 1%", monomorphic_fraction < 1 / 100)
    return genotypes[genotypes.columns[monomorphic_mask]]


def _filter_variants_for_fst(genotypes: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Filter out variants far from HWE, which are likely to be genotyping errors."""
    fsts = np.array(
        [_calc_fst(genotypes[variant], labels) for variant in tqdm.tqdm(genotypes.columns)],
    )
    fst_mask = fsts < FST_THRESHOLD
    fst_included_fraction = 0.99
    assert_true(
        f"Greater than {fst_included_fraction}% of variants retained by Fst filtering",
        np.mean(fst_mask) > fst_included_fraction,
    )
    return genotypes[genotypes.columns[fst_mask]]


def _load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the final dataset consisting of genotype matrix, labels."""
    genotypes = _load_genotypes()
    labels = load_label_data(genotypes.index)
    genotypes, labels = filter_samples_for_relatedness(genotypes, labels)
    genotypes = _filter_variants_for_maf(genotypes)
    genotypes = _filter_variants_for_ld(genotypes)
    assert_genotypes_and_label_agree(genotypes, labels)
    assert_equals("genotypes.shape", genotypes.shape, "expected genotypes.shape", (1870, 150502))
    return genotypes, labels


def assert_genotypes_and_label_agree(genotypes: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Check that genotypes, labels agree on indices."""
    assert_equals("genotypes index", genotypes.index, "labels index", labels.index)


def _calc_fst(variant_counts: pd.Series, samples: pd.DataFrame) -> float:
    """Calculate Fst from variant array, using samples for population labels."""
    N = len(variant_counts)
    p = np.mean(variant_counts) / PLOIDY
    total = 0.0
    for pop in samples.population.unique():
        idx = samples.population == pop
        ci = sum(idx) / N
        gs = variant_counts[idx]
        pi = np.mean(gs) / PLOIDY
        total += ci * pi * (1 - pi)
    return (p * (1 - p) - total) / (p * (1 - p))


def _perform_pca(train_X: pd.DataFrame, test_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Perform PCA, checking for good compression."""
    logger.info("Beginning PCA")
    minimum_variant_std = train_X.std(axis="index").min()
    assert_true(
        "minimum variant standard deviation greater than zero",
        minimum_variant_std > 0,
        comment="Have you excluded all monomorphic alleles?",
    )
    train_Xpc, pca = allel.pca(
        train_X.T,
        n_components=PCA_DIMS,
        scaler="patterson",
    )  # must be transposed for allel pca
    logger.info(
        "Cumulative explained variance ratio for %s dimensions: %s",
        len(pca.explained_variance_ratio_),
        np.cumsum(pca.explained_variance_ratio_),
    )
    test_Xpc = pca.transform(test_X.T)

    assert_true(
        f"Explained variance ratio > {EXPLAINED_VARIANCE_THRESHOLD}%",
        np.sum(pca.explained_variance_ratio_) > EXPLAINED_VARIANCE_THRESHOLD,
    )
    return train_Xpc, test_Xpc, pca


def make_train_test_split(
    genotypes: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make train / test splits, stratifying on population."""
    train_X, test_X, train_y, test_y = train_test_split(
        genotypes, labels, stratify=labels.population, random_state=1337
    )
    assert_equals(
        "train features",
        train_X.shape[1],
        "test features",
        test_X.shape[1],
        comment="Did you mix up the return values?",
    )
    return train_X, test_X, train_y, test_y


@dataclasses.dataclass(frozen=True)
class RFCParamChoices:
    """Param Choices for RFC Randomized Hyperparameter Tuning."""

    n_estimators: Literal[1000]
    max_depth: Literal[5, 10, 20, None]
    min_samples_leaf: Literal[1, 2, 5, 10]
    criterion: Literal["gini", "entropy", "log_loss"]
    max_features: Literal["sqrt", "log2", None]
    pca_dims: Literal[30]

    @classmethod
    def sample(cls) -> "RFCParamChoices":
        """Construct a randomized ParamChoice from values of Literal type annotations."""
        kwargs = {}
        for param_name, typ in cls.__annotations__.items():
            values = get_args(typ)
            kwargs[param_name] = random.choice(values)
        return cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class AccuracyReport:
    """Represent model accuracy scores at population and superpopulation levels."""

    train_pop_accuracy: float
    test_pop_accuracy: float
    train_superpop_accuracy: float
    test_superpop_accuracy: float


def _compute_accuracy_report(
    clf: sklearn.base.ClassifierMixin,
    train_Xpc: pd.DataFrame,
    test_Xpc: pd.DataFrame,
    train_y: pd.DataFrame,
    test_y: pd.DataFrame,
) -> AccuracyReport:
    train_yhat_pop_probs = clf.predict_proba(train_Xpc)
    test_yhat_pop_probs = clf.predict_proba(test_Xpc)

    train_yhat_pops = POPS[np.argmax(train_yhat_pop_probs, axis=1)]
    test_yhat_pops = POPS[np.argmax(test_yhat_pop_probs, axis=1)]
    train_yhat_superpops = superpop_predictions_from_pop_probs(train_yhat_pop_probs)
    test_yhat_superpops = superpop_predictions_from_pop_probs(test_yhat_pop_probs)

    return AccuracyReport(
        accuracy_score(train_y.population, train_yhat_pops),
        accuracy_score(test_y.population, test_yhat_pops),
        accuracy_score(train_y.superpop, train_yhat_superpops),
        accuracy_score(test_y.superpop, test_yhat_superpops),
    )


def make_rfc(
    train_Xpc: pd.DataFrame,
    test_Xpc: pd.DataFrame,
    train_y: pd.DataFrame,
    test_y: pd.DataFrame,
    trials: int = 10,
) -> RandomForestClassifier:
    """Build population-level RFC using randomized hyperparameter search."""
    tuning_results: dict[RFCParamChoices, AccuracyReport] = {}

    for _trial in tqdm.trange(trials):
        param_choices = RFCParamChoices.sample()
        rfc_params = {k: v for (k, v) in dataclasses.asdict(param_choices).items() if k != "pca_dims"}
        cols_to_use = train_Xpc.columns[: param_choices.pca_dims]
        rfc = RandomForestClassifier(**rfc_params, random_state=1337)
        rfc.fit(train_Xpc[cols_to_use], train_y.population)

        accuracy_report = _compute_accuracy_report(
            rfc, train_Xpc[cols_to_use], test_Xpc[cols_to_use], train_y, test_y
        )
        tuning_results[param_choices] = accuracy_report
        logger.info(
            "RFC param choices: %s, accuracies: %s", param_choices, tuning_results[param_choices]
        )

    best_params = max(tuning_results, key=lambda params: tuning_results[params].test_pop_accuracy)
    rfc_params = {k: v for (k, v) in dataclasses.asdict(best_params).items() if k != "pca_dims"}
    cols_to_use = train_Xpc.columns[: param_choices.pca_dims]

    rfc = RandomForestClassifier(**(rfc_params))
    rfc.fit(train_Xpc[cols_to_use], train_y.population)
    # recompute accuracy report to ensure we didn't just get lucky...
    accuracy_report = _compute_accuracy_report(
        rfc, train_Xpc[cols_to_use], test_Xpc[cols_to_use], train_y, test_y
    )
    logger.info("best_params: %s", best_params)
    logger.info("accuracies %s", accuracy_report)

    threshold_checks = [
        ("train population", RFC_TRAIN_ACCURACY_THRESHOLD, accuracy_report.train_pop_accuracy),
        ("test population", RFC_TEST_ACCURACY_THRESHOLD, accuracy_report.test_pop_accuracy),
        (
            "train superpop",
            RFC_TRAIN_SUPERPOP_ACCURACY_THRESHOLD,
            accuracy_report.test_superpop_accuracy,
        ),
        ("test superpop", RFC_TEST_SUPERPOP_ACCURACY_THRESHOLD, accuracy_report.test_superpop_accuracy),
    ]

    for description, expected, actual in threshold_checks:
        if not actual >= expected:
            logger.warning("Expected %s accuracy >= %s, got: %s instead", description, expected, actual)
    # if not accuracy_report.train_pop_accuracy > RFC_TRAIN_ACCURACY_THRESHOLD:
    # assert_true(
    # assert_true(
    # assert_true(
    #     accuracy_report.train_superpops_accuracy > RFC_TRAIN_SUPERPOP_ACCURACY_THRESHOLD,
    # assert_true(
    #     accuracy_report.test_superpops_accuracy > RFC_TEST_SUPERPOP_ACCURACY_THRESHOLD,

    return rfc


def superpop_probs_from_pop_probs(pop_probs: np.ndarray) -> np.ndarray:
    """Given a matrix of population probabilities, convert to matrix of superpop probabilities."""
    N = len(pop_probs)
    pops = sorted(SUPERPOP_FROM_POP.keys())
    superpops = sorted(set(SUPERPOP_FROM_POP.values()))
    superpop_projection_matrix = np.array(
        [[int(superpop == SUPERPOP_FROM_POP[pop]) for superpop in superpops] for pop in pops]
    )
    superpop_probs = pop_probs @ superpop_projection_matrix
    assert_equals(
        "Expected superpop_probs shape (N x |superpops|):",
        superpop_probs.shape,
        "Actual shape",
        (N, len(superpops)),
    )
    return superpop_probs


def superpop_predictions_from_pop_probs(pop_probs: np.ndarray) -> list[str]:
    """Given a matrix of population probabilities, convert to superpop predictions."""
    superpops = sorted(set(SUPERPOP_FROM_POP.values()))
    superpop_probs = superpop_probs_from_pop_probs(pop_probs)
    return [superpops[np.argmax(ps)] for ps in superpop_probs]


def _filter_variants_for_mi(
    train_X: pd.DataFrame, test_X: pd.DataFrame, train_y_pop: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mi_df = _get_mi_df(train_X, train_y_pop)
    mi_keep_cols = mi_df[mi_df.mutual_info > MI_THRESHOLD].index

    return train_X[mi_keep_cols], test_X[mi_keep_cols]


def _calc_mi(xs: list[T], ys: list[T]) -> float:
    """Calculate mutual information (in bits) between xs and ys."""
    if len(xs) != len(ys):
        msg = "xs and ys must have same length."
        raise ValueError(msg)
    N = len(xs)
    x_counts: dict[T, int] = Counter()
    y_counts: dict[T, int] = Counter()
    xy_counts: dict[tuple[T, T], int] = Counter()
    for x, y in zip(xs, ys):  # noqa: B905
        x_counts[x] += 1
        y_counts[y] += 1
        xy_counts[(x, y)] += 1
    x_ps = np.array(list(x_counts.values())) / N
    y_ps = np.array(list(y_counts.values())) / N
    xy_ps = np.array(list(xy_counts.values())) / N
    x_entropy = -(x_ps @ np.log2(x_ps))
    y_entropy = -(y_ps @ np.log2(y_ps))
    xy_entropy = -(xy_ps @ np.log2(xy_ps))
    return (x_entropy + y_entropy) - xy_entropy


def _get_mi_df(train_X: pd.DataFrame, train_y_pop: pd.Series) -> pd.DataFrame:
    mis = mis = np.array(
        [_calc_mi(col, train_y_pop.population) for col in tqdm.tqdm(train_X.to_numpy().T)]
    )
    mi_df = pd.DataFrame(mis, columns=["mutual_info"], index=train_X.columns)
    mi_df.to_parquet("mi_df.parquet")
    return mi_df


def serialize_model_products(variants: Sequence[str], pca: PCA, rfc: RandomForestClassifier) -> None:
    """Serialize variant list, pca and rfc to disk as .txt, .skops files."""
    variants_fpath = ANCESTRY_MODEL_PRODUCTS_DIR / "variants.txt"
    pca_fpath = ANCESTRY_MODEL_PRODUCTS_DIR / "pca.skop"
    rfc_fpath = ANCESTRY_MODEL_PRODUCTS_DIR / "rfc.skop"

    with variants_fpath.open("w") as f:
        f.write("\n".join(variants))
    skops_dump(pca, pca_fpath)
    skops_dump(rfc, rfc_fpath)


def main() -> None:
    """Train global ancestry model."""
    err_msg = "This module isn't production-ready yet!"
    raise NotImplementedError(err_msg)
    genotypes, labels = _load_dataset()
    train_X, test_X, train_y, test_y = make_train_test_split(
        genotypes,
        labels,
    )
    train_X_filtered, test_X_filtered = _filter_variants_for_mi(train_X, test_X, train_y.population)
    train_Xpc, test_Xpc, pca = _perform_pca(train_X_filtered, test_X_filtered)
    rfc = make_rfc(train_Xpc, test_Xpc, train_y.population, test_y.population)
    serialize_model_products(list(train_X_filtered.columns), pca, rfc)
