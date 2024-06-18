"""Load and prep fragpipe data-indepdent analysis (DIA) datasets."""

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
from bystro.proteomics.fragpipe_utils import check_df_starts_with_cols, prep_annotation_df

pd.options.future.infer_string = True  # type: ignore

PG_MATRIX_COLS = ["Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"]


@dataclass(frozen=True)
class DataIndependentAnalysisDataset:
    """Represent a Fragpipe Tandem Mass Tag dataset."""

    pg_matrix_df: pd.DataFrame
    annotation_df: pd.DataFrame


def _prep_pg_matrix_df(pg_matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Prep pg_matrix_df, setting Protein.IDs as index and dropping extraneous columns."""
    check_df_starts_with_cols(pg_matrix_df, PG_MATRIX_COLS)
    pg_matrix_df = pg_matrix_df.set_index("Protein.Ids")
    pg_matrix_df = pg_matrix_df.drop(
        ["Protein.Group", "Protein.Names", "Genes", "First.Protein.Description"], axis="columns"
    )
    return pg_matrix_df


def load_data_independent_analysis_dataset(
    pg_matrix_filename: Path | str | StringIO, annotation_filename: Path | str | StringIO
) -> DataIndependentAnalysisDataset:
    """Load and prep Fragpipe tandem mass tag datasets."""
    raw_pg_matrix_df = pd.read_csv(pg_matrix_filename, sep="\t")
    raw_annotation_df = pd.read_csv(annotation_filename, sep="\t")
    pg_matrix_df = _prep_pg_matrix_df(raw_pg_matrix_df)
    annotation_df = prep_annotation_df(raw_annotation_df)
    return DataIndependentAnalysisDataset(pg_matrix_df, annotation_df)
