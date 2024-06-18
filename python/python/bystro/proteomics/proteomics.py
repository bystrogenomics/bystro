"""Load and preprocess fragpipe output files."""

from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import TypeVar

import pandas as pd

pd.options.future.infer_string = True  # type: ignore

T = TypeVar("T")


def load_fragpipe_dataset(fname: str | Path | StringIO) -> pd.DataFrame:
    """Load a fragpipe dataset and prep it according to type of dataset."""
    return _prep_fragpipe_dataset(_load_fragpipe_tsv(fname))


# ------------------------------- END PUBLIC API -------------------------------


def _load_fragpipe_tsv(fname: str | Path | StringIO) -> pd.DataFrame:
    """Read a fragpipe dataset from disk."""
    return pd.read_csv(fname, sep="\t", index_col="Index")


def _list_startswith(xs: Sequence[T], ys: Sequence[T]) -> bool:
    """Determine whether list xs starts with list ys."""
    return xs[: len(ys)] == ys


def _prep_fragpipe_dataset(fragpipe_dataset: pd.DataFrame) -> pd.DataFrame:
    """Recognize dataset type and dispatch appropriate preprocessing function."""
    type1_columns = ["Gene", "Peptide", "ReferenceIntensity"]
    type2_columns = ["NumberPSM", "Proteins", "ReferenceIntensity"]
    actual_columns = list(fragpipe_dataset.columns)
    if _list_startswith(actual_columns, type1_columns):
        fragpipe_df = _prep_fragpipe_dataset_type1(fragpipe_dataset)
    elif _list_startswith(actual_columns, type2_columns):
        fragpipe_df = _prep_fragpipe_dataset_type2(fragpipe_dataset)
    else:
        err_msg = (
            f"Dataset format not recognized: "
            f"expected columns to begin with {type1_columns} or {type2_columns}"
        )
        raise ValueError(err_msg)
    fragpipe_df.columns.name = "gene"
    fragpipe_df.index.name = "sample"
    return fragpipe_df


def _subtract_reference_intensities(fragpipe_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe by reference intensities."""
    reference_intensities = fragpipe_df["ReferenceIntensity"]
    fragpipe_df = fragpipe_df.subtract(reference_intensities, axis=0)
    fragpipe_df = fragpipe_df.drop(["ReferenceIntensity"], axis=1)
    return fragpipe_df


def _prep_fragpipe_dataset_type1(fragpipe_df: pd.DataFrame) -> pd.DataFrame:
    """Prep fragpipe dataset where multiple peptides map into gene."""
    fragpipe_df = fragpipe_df.drop(["Peptide"], axis="columns")
    fragpipe_df = fragpipe_df.groupby("Gene").apply(
        lambda g: g.mean(axis="index")
    )  # average over all peptides belonging to gene
    fragpipe_df = _subtract_reference_intensities(fragpipe_df)
    fragpipe_df = fragpipe_df.T  # convert to (samples X genes)
    return fragpipe_df


def _prep_fragpipe_dataset_type2(fragpipe_df: pd.DataFrame) -> pd.DataFrame:
    """Prep fragpipe dataset where peptide aggregation has already been performed."""
    fragpipe_df = fragpipe_df.drop(["NumberPSM", "Proteins"], axis="columns")
    fragpipe_df = _subtract_reference_intensities(fragpipe_df)
    fragpipe_df = fragpipe_df.T  # convert to (samples X genes)
    return fragpipe_df
