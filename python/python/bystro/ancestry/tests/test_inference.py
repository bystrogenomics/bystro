"""Test Ancestry Model inference code."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier  # type: ignore

import pyarrow as pa  # type: ignore
import pyarrow.dataset as ds  # type: ignore

from bystro.ancestry.inference import AncestryModel, AncestryModels, infer_ancestry
from bystro.ancestry.train import POPS
from bystro.ancestry.model import get_models

pd.options.future.infer_string = True # type: ignore

SAMPLES = [f"sample{i}" for i in range(len(POPS))]
VARIANTS = ["variant1", "variant2", "variant3"]
PC_COLUMNS = ["pc1", "pc2", "pc3", "pc4"]
FAKE_GENOTYPES = pd.DataFrame(
    np.random.random((len(VARIANTS), len(SAMPLES))), index=VARIANTS, columns=SAMPLES
)
FAKE_GENOTYPES_DOSAGE_MATRIX = ds.dataset(
    pa.Table.from_pandas(
        FAKE_GENOTYPES.reset_index().rename(columns={"index": "locus"}), preserve_index=False
    ).to_batches()
)


def _infer_ancestry():
    samples = [f"sample{i}" for i in range(len(POPS))]
    variants = ["variant1", "variant2", "variant3"]
    pc_columns = ["pc1", "pc2", "pc3", "pc4"]
    pca_loadings_df = pd.DataFrame(
        np.random.random((len(variants), len(pc_columns))), index=variants, columns=pc_columns
    )
    genotypes = pd.DataFrame(
        np.random.random((len(variants), len(samples))), index=variants, columns=samples
    )

    train_Xpc = genotypes.T @ pca_loadings_df
    train_y = POPS
    rfc = RandomForestClassifier(n_estimators=1, max_depth=1).fit(train_Xpc, train_y)
    ancestry_model = AncestryModel(pca_loadings_df, rfc)

    genotypes = genotypes.reset_index()
    genotypes = genotypes.rename(columns={"index": "locus"})
    genotypes = ds.dataset(pa.Table.from_pandas(genotypes, preserve_index=False).to_batches())

    return infer_ancestry(AncestryModels(ancestry_model, ancestry_model), genotypes), samples


def _make_ancestry_model() -> AncestryModel:
    # one population per sample so that we can include all populations in train_y.
    pca_loadings_df = pd.DataFrame(
        np.random.random((len(VARIANTS), len(PC_COLUMNS))), index=VARIANTS, columns=PC_COLUMNS
    )
    train_Xpc = FAKE_GENOTYPES.T @ pca_loadings_df
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
    ancestry_response, samples = _infer_ancestry()

    assert len(samples) == len(ancestry_response.results)


def test_infer_ancestry_from_model():
    ancestry_models = get_models("hg38")

    # Generate an arrow table that contains genotype dosages for 1000 samples
    variants = list(ancestry_models.gnomad_model.pca_loadings_df.index)
    samples = [f"sample{i}" for i in range(1000)]
    genotypes = pd.DataFrame(
        np.random.randint(0, 2, (len(variants), len(samples))),  # noqa: NPY002
        index=variants,
        columns=samples,  # noqa: NPY002
    )
    # randomly set 10% of the genotypes to missing to ensure we test missing data handling
    drop_snps_n = int(0.1 * len(genotypes))
    retained_snps_n = len(genotypes) - drop_snps_n
    drop_indices = np.random.choice(genotypes.index, size=drop_snps_n, replace=False) # noqa: NPY002
    genotypes = genotypes.drop(list(drop_indices))

    genotypes = genotypes.reset_index()
    genotypes = genotypes.rename(columns={"index": "locus"})

    genotypes = ds.dataset(pa.Table.from_pandas(genotypes, preserve_index=False).to_batches())

    ancestry_response = infer_ancestry(ancestry_models, genotypes)

    assert len(samples) == len(ancestry_response.results)

    top_hits = set()
    top_probs = set()
    samples_seen = set()
    sample_set = set(samples)
    for result in ancestry_response.results:
        top_hits.add(result.top_hit.populations[0])
        top_probs.add(result.top_hit.probability)

        samples_seen.add(result.sample_id)

        assert result.n_snps == retained_snps_n

    assert samples_seen == sample_set
    assert len(top_hits) > 1
    assert len(top_probs) > 1
