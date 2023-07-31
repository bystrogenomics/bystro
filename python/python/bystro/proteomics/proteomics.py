"""Load and postprocess Fragpipe output files."""


from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import pandas as pd


def _load_fragpipe_tsv(fname: str | Path) -> pd.DataFrame:
    return pd.read_csv(fname, sep="\t", index_col="Index")


def load_fragpipe_dataset(fname: str | Path) -> pd.DataFrame:
    """Load a fragpipe dataset and transform it according to columns present."""
    fragpipe_dataset = _load_fragpipe_tsv(fname)
    return _transform_fragpipe_dataset(fragpipe_dataset)


T = TypeVar("T")


def _list_startswith(xs: Sequence[T], ys: Sequence[T]) -> bool:
    """Determine whether list xs starts with list ys."""
    return xs[: len(ys)] == ys


def _transform_fragpipe_dataset(fragpipe_dataset: pd.DataFrame) -> pd.DataFrame:
    type1_columns = ["Gene", "Peptide", "ReferenceIntensity"]
    type2_columns = ["NumberPSM", "Proteins", "ReferenceIntensity"]
    actual_columns = list(fragpipe_dataset.columns)
    if _list_startswith(actual_columns, type1_columns):
        fragpipe_df = _transform_fragpipe_dataset_type1(fragpipe_dataset)
    elif _list_startswith(actual_columns, type2_columns):
        fragpipe_df = _transform_fragpipe_dataset_type2(fragpipe_dataset)
    else:
        err_msg = (
            f"Dataset format not recognized: "
            f"expected columns to begin with {type1_columns} or {type2_columns}"
        )
        raise ValueError(err_msg)
    fragpipe_df.columns.name = "gene"
    fragpipe_df.index.name = "sample"
    return fragpipe_df


def _transform_fragpipe_dataset_type1(fragpipe_df: pd.DataFrame) -> pd.DataFrame:
    fragpipe_df = fragpipe_df.drop(["ReferenceIntensity", "Peptide"], axis="columns")
    fragpipe_df = fragpipe_df.groupby("Gene").apply(
        lambda g: g.mean(axis="index")
    )  # average over all peptides belonging to gene
    fragpipe_df = fragpipe_df.T  # convert to (samples X genes)
    return fragpipe_df


def _transform_fragpipe_dataset_type2(fragpipe_df: pd.DataFrame) -> pd.DataFrame:
    fragpipe_df = fragpipe_df.drop(["NumberPSM", "Proteins", "ReferenceIntensity"], axis="columns")
    fragpipe_df = fragpipe_df.T  # convert to (samples X genes)
    return fragpipe_df


fragpipe_directory = Path("/Users/patrickoneil/wingolab-bystro-matrixData-opensearch/data/fragpipe")

filenames = [
    fragpipe_directory / f
    for f in [
        "Supplementary_Data_Proteome_DIA/6_CPTAC3_CCRCC_Whole_abundance_gene_protNorm=2_CB.tsv",
        "Supplementary_Data_Phosphoproteome_DIA/6_CPTAC3_CCRCC_Phospho_abundance_phosphosite_protNorm=2_CB_imputed.tsv",
        "Supplementary_Data_Phosphoproteome_DIA/6_CPTAC3_CCRCC_Phospho_abundance_gene_protNorm=2_CB_imputed.tsv",
        "Supplementary_Data_Phosphoproteome_DIA/6_CPTAC3_CCRCC_Phospho_abundance_phosphopeptide_protNorm=2_CB_1211.tsv",
        "Supplementary_Data_Phosphoproteome_DIA/6_CPTAC3_CCRCC_Phospho_abundance_phosphopeptide_protNorm=2_CB_imputed_1211.tsv",
    ]
]
