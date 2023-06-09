"""Training code for Global Ancestry Model.

This script takes a vcf of preprocessed variants (see preprocess_vcfs.sh) and generates:

1.  A list of variants to be used for inference.

2.  PCA loadings mapping the list of variants in (1) to PC space.

3.  Classifiers mapping PC space to the 26 HapMap populations as well as 5 continent-level
superpopulations.
"""

import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import allel
import numpy as np
import pandas as pd
import skops
import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bystro.ancestry.asserts import assert_equals, assert_true
from bystro.ancestry.train_utils import get_variant_ids_from_callset

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATA_ROOT_DIR = Path.home() / "emory/human_ancestry_project/data/1000_genome_VCF"
VCF_PATH = DATA_ROOT_DIR / "1KGP_final_variants_1percent.vcf.gz"
ANCESTRY_MODEL_PRODUCTS_DIR = Path("ancestry_model_products")

ANCESTRY_INFO_PATH = DATA_ROOT_DIR / "20130606_sample_info.txt"
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
RFC_TEST_ACCURACY_THRESHOLD = 0.8
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
    "MAG": "AFR",
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


rng = np.random.RandomState(1337)


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


def _filter_samples_for_relatedness(
    genotypes: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter samples for relatedness, returning subset of unrelated individuals."""
    logger.info("Filtering samples for relatedness")
    ancestry_df = _load_ancestry_df()
    assert_equals("genotype samples", genotypes.index, "label samples", labels.index)
    samples = genotypes.index
    ancestry_df = ancestry_df[ancestry_df["Sample"].isin(samples)]
    family_ids = ancestry_df["Family ID"].unique()
    logger.info("Found %s unique families", len(family_ids))
    unrelated_samples = []
    for family_id in family_ids:
        family_members = ancestry_df["Sample"][ancestry_df["Family ID"] == family_id]
        # grab value out of array...
        family_member = family_members.sample(1, random_state=rng).to_numpy()[0]
        unrelated_samples.append(family_member)
    unrelated_sample_idx = np.array(unrelated_samples)
    genotypes, labels = genotypes.loc[unrelated_sample_idx], labels.loc[unrelated_sample_idx]
    # we did this earlier but removing samples could make more variants monomorphic
    genotypes = _filter_variants_for_monomorphism(genotypes)
    return genotypes, labels


def _load_ancestry_df() -> pd.DataFrame:
    return pd.read_csv(ANCESTRY_INFO_PATH, sep="\t")


def _load_label_data(samples: pd.Index) -> pd.DataFrame:
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
    labels = _load_label_data(genotypes.index)
    genotypes, labels = _filter_samples_for_relatedness(genotypes, labels)
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


def _perform_pca(train_X: np.ndarray, test_X: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Perform PCA, checking for good compression."""
    logger.info("Beginning PCA")
    minimum_variant_std = train_X.std(axis=ROWS).min()
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


def _make_train_test_split(
    genotypes: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make train / test splits, stratifying on population."""
    train_X, test_X, train_y, test_y = train_test_split(
        genotypes,
        labels,
        stratify=labels.population,
    )
    assert_equals(
        "train features",
        train_X.shape[1],
        "test features",
        test_X.shape[1],
        comment="Did you mix up the return values?",
    )
    return train_X, test_X, train_y, test_y


def _make_rfc(
    train_Xpc: np.ndarray,
    test_Xpc: np.ndarray,
    train_y: pd.Series,
    test_y: pd.Series,
) -> RandomForestClassifier:
    trials = 10
    tuning_results = {}

    param_choices = {
        "n_estimators": [1000],
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_features": ["sqrt", "log2", None],
    }

    for _trial in range(trials):
        params: dict[str, str | int | None] = {k: random.choice(vs) for (k, vs) in param_choices.items()}
        rfc = RandomForestClassifier(**params.items(), random_state=1337)
        rfc.fit(train_Xpc, train_y.population)
        train_yhat = rfc.predict(train_Xpc)
        test_yhat = rfc.predict(test_Xpc)
        train_acc, test_acc = (
            accuracy_score(train_y.population, train_yhat),
            accuracy_score(test_y.population, test_yhat),
        )
        param_tuple = tuple(sorted(params.items()))
        tuning_results[param_tuple] = (train_acc, test_acc)

    values, test_accuracy = 1, 1
    best_params = max(tuning_results.items(), key=lambda kv: kv[values][test_accuracy])
    rfc = RandomForestClassifier(**{k: v for (k, v) in best_params.items() if k != "pca_dim"})
    rfc.fit(train_Xpc, train_y.population)
    train_yhat = rfc.predict(train_Xpc)
    test_yhat = rfc.predict(test_Xpc)
    train_acc, test_acc = (
        accuracy_score(train_y.population, train_yhat),
        accuracy_score(test_y.population, test_yhat),
    )
    assert_true("RFC train accuracy >= 0.9", train_acc > RFC_TRAIN_ACCURACY_THRESHOLD)
    assert_true("RFC test accuracy >= 0.8", test_acc > RFC_TEST_ACCURACY_THRESHOLD)


def _filter_variants_for_mi(
    train_X: pd.DataFrame, test_X: pd.DataFrame, train_y_pop: pd.Series
) -> np.ndarray:
    mi_df = _get_mi_df(train_X, train_y_pop)
    mi_keep_cols = mi_df[mi_df.mutual_info > MI_THRESHOLD].index

    return train_X[mi_keep_cols], test_X[mi_keep_cols]


def _calc_mi(xs: list, ys: list) -> float:
    """Calculate mutual information between xs and ys."""
    if len(xs) != len(ys):
        msg = "xs and ys must have same length."
        raise ValueError(msg)
    N = len(xs)
    x_counts = Counter()
    y_counts = Counter()
    xy_counts = Counter()
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


def _get_mi_df(train_X: np.ndarray, train_y_pop: pd.Series) -> np.ndarray:
    mis = mis = np.array(
        [_calc_mi(col, train_y_pop.population) for col in tqdm.tqdm(train_X.to_numpy().T)]
    )
    mi_df = pd.DataFrame(mis, columns=["mutual_info"], index=train_X.columns)
    mi_df.to_parquet("mi_df.parquet")
    return mi_df


def _serialize_data(variants: list[str], pca: PCA, rfc: RandomForestClassifier) -> None:
    variants_fpath = ANCESTRY_MODEL_PRODUCTS_DIR / "variants.txt"
    pca_fpath = ANCESTRY_MODEL_PRODUCTS_DIR / "pca.skop"
    rfc_fpath = ANCESTRY_MODEL_PRODUCTS_DIR / "rfc.skop"

    with variants_fpath.open("w") as f:
        f.write("\n".join(variants))
    skops.io.dump(pca, pca_fpath)
    skops.io.dump(rfc, rfc_fpath)


def main() -> None:
    """Train global ancestry model."""
    genotypes, labels = _load_dataset()
    train_X, test_X, train_y, test_y = _make_train_test_split(
        genotypes,
        labels,
    )
    train_X_filtered, test_X_filtered = _filter_variants_for_mi(train_X, test_X, train_y.population)
    train_Xpc, test_Xpc, pca = _perform_pca(train_X_filtered, test_X_filtered)
    rfc = _make_rfc(train_Xpc, test_Xpc, train_y.population, test_y.population)
    _serialize_data(train_X_filtered.columns, pca, rfc)
