"""Test Ancestry Model inference code."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from bystro.ancestry.inference import AncestryModel, _fill_missing_data, infer_ancestry
from bystro.ancestry.train import POPS

SAMPLES = [f"sample{i}" for i in range(len(POPS))]
VARIANTS = ["variant1", "variant2", "variant3"]
PC_COLUMNS = ["pc1", "pc2", "pc3", "pc4"]
FAKE_GENOTYPES = pd.DataFrame(
    np.random.random((len(SAMPLES), len(VARIANTS))), index=SAMPLES, columns=VARIANTS
)


def _make_ancestry_model() -> AncestryModel:
    # one population per sample so that we can include all populations in train_y.
    pca_loadings_df = pd.DataFrame(
        np.random.random((len(VARIANTS), len(PC_COLUMNS))), index=VARIANTS, columns=PC_COLUMNS
    )
    train_Xpc = FAKE_GENOTYPES @ pca_loadings_df
    train_y = POPS
    rfc = RandomForestClassifier(n_estimators=1, max_depth=1).fit(train_Xpc, train_y)
    return AncestryModel(pca_loadings_df, rfc)


ANCESTRY_MODEL = _make_ancestry_model()


def test_Ancestry_Model():
    pcs_for_plotting, pop_probs = ANCESTRY_MODEL.predict_proba(FAKE_GENOTYPES)
    assert (pop_probs.index == SAMPLES).all()
    assert (pop_probs.columns == POPS).all()


def test_Ancestry_Model_missing_pca_col():
    pca_loadings_df = ANCESTRY_MODEL.pca_loadings_df
    bad_pca_loadings_df = pca_loadings_df[pca_loadings_df.columns[:-1]]

    with pytest.raises(ValueError, match="must equal"):
        AncestryModel(bad_pca_loadings_df, ANCESTRY_MODEL.rfc)


def test_infer_ancestry():
    samples = [f"sample{i}" for i in range(len(POPS))]  # one pop per sample
    variants = ["variant1", "variant2", "variant3"]
    pc_columns = ["pc1", "pc2", "pc3", "pc4"]
    pca_loadings_df = pd.DataFrame(
        np.random.random((len(variants), len(pc_columns))), index=variants, columns=pc_columns
    )
    train_X = pd.DataFrame(
        np.random.random((len(samples), len(variants))), index=samples, columns=variants
    )
    train_Xpc = train_X @ pca_loadings_df
    train_y = POPS
    rfc = RandomForestClassifier(n_estimators=1, max_depth=1).fit(train_Xpc, train_y)
    ancestry_model = AncestryModel(pca_loadings_df, rfc)
    vcf_path = "my_vcf.vcf"
    ancestry_response = infer_ancestry(ancestry_model, train_X, vcf_path)
    assert len(samples) == len(ancestry_response.results)
    assert vcf_path == ancestry_response.vcf_path


def test__fill_missing_data():
    genotypes = pd.DataFrame(np.random.random((3, 3)))
    genotypes.iloc[0, 0] = np.nan
    genotypes.iloc[1, 2] = np.nan
    genotypes.iloc[2, 1] = np.nan

    filled_genotypes, missingnesses = _fill_missing_data(genotypes)
    assert filled_genotypes.notna().all().all()
    assert np.allclose(genotypes.mean(), filled_genotypes.mean())
    assert (missingnesses == 1 / 3).all()


def test__fill_missing_data_col_completely_nan():
    genotypes = pd.DataFrame(np.random.random((3, 3)))
    genotypes.iloc[:, 0] = np.nan

    filled_genotypes, missingnesses = _fill_missing_data(genotypes)
    assert not filled_genotypes.isna().any().any()
