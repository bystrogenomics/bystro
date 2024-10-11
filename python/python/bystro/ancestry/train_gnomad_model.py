"""Implement PCA + RF for variants gnomad uses for ancestry."""

import pandas as pd
from sklearn.decomposition import PCA

from bystro.ancestry.train import (
    load_callset_for_variants,
    load_label_data,
    make_rfc,
    make_train_test_split,
    serialize_model_products,
)

VARIANT_PATH = "hg38_gnomad_snpset.csv"

pd.options.future.infer_string = True  # type: ignore


def load_model_variants() -> set[str]:
    """Get set of variants to train ancestry model."""
    try:
        variants_df = pd.read_csv(VARIANT_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Variant file '{VARIANT_PATH}' not found.")
    except pd.errors.ParserError as e:
        raise Exception(f"Error parsing variant file '{VARIANT_PATH}': {e}")
    return set(variants_df.variants)


def load_kgp_genotypes_for_shared_variants() -> pd.DataFrame:
    """Get KGP genotypes filtered to shared variants."""
    variants = load_model_variants()
    return load_callset_for_variants(variants)


def main() -> None:
    """Train PCA, RF for gnomad variants, save model products to disk."""
    kgp_genotypes = load_kgp_genotypes_for_shared_variants()
    labels = load_label_data(kgp_genotypes.index)
    train_X, test_X, train_y, test_y = make_train_test_split(
        kgp_genotypes,
        labels,
    )
    PCA_DIMS = 30
    pca = PCA(n_components=PCA_DIMS).fit(train_X)
    pc_columns = [f"pc{i}" for i in range(1, PCA_DIMS + 1)]
    loadings_df = pd.DataFrame(pca.components_.T, index=train_X.columns, columns=pc_columns)
    train_Xpc = train_X @ loadings_df
    test_Xpc = test_X @ loadings_df
    rfc = make_rfc(train_Xpc, test_Xpc, train_y, test_y)
    serialize_model_products(loadings_df, rfc)


if __name__ == "__main__":
    main()
