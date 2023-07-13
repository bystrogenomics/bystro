"""Implement PCA + RF for shared Illumina + Affymetrix variants."""

import pandas as pd
from sklearn.decomposition import PCA
from train import (
    filter_samples_for_relatedness,
    load_callset_for_variants,
    load_label_data,
    make_rfc,
    make_train_test_split,
    serialize_model_products,
)


def load_illumina_affy_variants() -> set(str):
    """Get previously computed intersection of illumina and affymetrix variants."""
    return set(pd.read_csv("shared_illumina_affy_variants.csv").variant)


def load_kgp_genotypes_for_shared_variants() -> pd.DataFrame:
    """Get KGP genotypes filtered to shared variants."""
    variants = load_illumina_affy_variants()
    return load_callset_for_variants(variants)


def main() -> None:
    """Train PCA, RF for Illumina and Affymetrix variants, save model products to disk."""
    kgp_genotypes = load_kgp_genotypes_for_shared_variants()
    labels = load_label_data(kgp_genotypes.index)
    kgp_genotypes, labels = filter_samples_for_relatedness(kgp_genotypes, labels)
    train_X, test_X, train_y, test_y = make_train_test_split(
        kgp_genotypes,
        labels,
    )
    PCA_DIMS = 30
    pca = PCA(n_components=PCA_DIMS).fit(train_X)
    train_Xpc = pd.DataFrame(
        pca.transform(train_X),
        columns=["pc" + str(i) for i in range(1, PCA_DIMS + 1)],
        index=train_X.index,
    )
    test_Xpc = pd.DataFrame(
        pca.transform(test_X),
        columns=["pc" + str(i) for i in range(1, PCA_DIMS + 1)],
        index=test_X.index,
    )
    rfc = make_rfc(train_Xpc, test_Xpc, train_y, test_y)
    serialize_model_products(kgp_genotypes.index, pca, rfc)
