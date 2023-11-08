"""Load and prep fragpipe tandem mass Tag datasets."""

from dataclasses import dataclass
from io import StringIO

import pandas as pd
from bystro.proteomics.fragpipe_utils import check_df_starts_with_cols, prep_annotation_df

ABUNDANCE_COLS = ["Index", "NumberPSM", "ProteinID", "MaxPepProb", "ReferenceIntensity"]


@dataclass(frozen=True)
class TandemMassTagDataset:
    """Represent a Fragpipe Tandem Mass Tag dataset."""

    abundance_df: pd.DataFrame
    annotation_df: pd.DataFrame

    def __post_init__(self) -> None:
        try:
            check_df_starts_with_cols(self.abundance_df, ABUNDANCE_COLS[1:])
        except ValueError as e:
            err_msg = "Received abundance_df with unexpected columns"
            raise ValueError(err_msg) from e

    def get_melted_abundance_df(self) -> pd.DataFrame:
        """Return a melted abundance df with columns [gene_name, sample_id, value]"""
        abundance_df = self.abundance_df
        columns_to_drop = ["NumberPSM", "ProteinID", "MaxPepProb", "ReferenceIntensity"]
        final_column_ordering = ["sample_id", "gene_name", "value"]
        melted_df_with_unsorted_columns = (
            abundance_df.drop(columns=columns_to_drop)
            .melt(var_name=["sample_id"], ignore_index=False)  # type: ignore[arg-type]
            .reset_index(names="gene_name")
        )
        return melted_df_with_unsorted_columns[final_column_ordering]


def _prep_abundance_df(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """Prep abundance_df, setting index and normalizing abundances by ReferenceIntensity."""
    check_df_starts_with_cols(abundance_df, ABUNDANCE_COLS)
    abundance_df = abundance_df.set_index("Index")
    first_sample_column_idx = abundance_df.columns.to_list().index("ReferenceIntensity") + 1
    sample_columns = abundance_df.columns[first_sample_column_idx:]
    for sample_column in sample_columns:
        abundance_df[sample_column] -= abundance_df.ReferenceIntensity
    return abundance_df


def load_tandem_mass_tag_dataset(
    abundance_filename: str | StringIO, annotation_filename: str | StringIO
) -> TandemMassTagDataset:
    """Load and prep Fragpipe tandem mass tag datasets."""
    raw_abundance_df = pd.read_csv(abundance_filename, sep="\t")
    raw_annotation_df = pd.read_csv(annotation_filename, sep="\t")
    abundance_df = _prep_abundance_df(raw_abundance_df)
    annotation_df = prep_annotation_df(raw_annotation_df)
    return TandemMassTagDataset(abundance_df, annotation_df)
