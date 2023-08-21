"""Load and prep fragpipe tandem mass Tag datasets."""

from dataclasses import dataclass

import pandas as pd
from bystro.proteomics.fragpipe_utils import prep_annotation_df

PG_MATRIX_COLS = ["Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"]


@dataclass(frozen=True)
class DataIndependentAnalysisDataset:
    """Represent a Fragpipe Tandem Mass Tag dataset."""

    pg_matrix_df: pd.DataFrame
    annotation_df: pd.DataFrame


def _prep_pg_matrix_df(pg_matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Prep pg_matrix_df, setting index and normalizing pg_matrixs by ReferenceIntensity."""
    _check_df_cols(pg_matrix_df, PG_MATRIX_COLS)
    pg_matrix_df = pg_matrix_df.set_index("Protein.Ids")
    pg_matrix_df = pg_matrix_df.drop(
        ["Protein.Group", "Protein.Names", "Genes", "First.Protein.Description"], axis="columns"
    )
    return pg_matrix_df


def load_data_independent_analysis_dataset(
    abundance_filename: str, annotation_filename: str
) -> DataIndependentAnalysisDataset:
    """Load and prep Fragpipe tandem mass tag datasets."""
    raw_abundance_df = pd.read_csv(pg_matrix_filename, sep="\t")
    raw_annotation_df = pd.read_csv(annotation_filename, sep="\t")
    abundance_df = _prep_abundance_df(raw_abundance_df)
    annotation_df = prep_annotation_df(raw_annotation_df)
    return DataIndependentAnalysisDataset(abundance_df, annotation_df)
