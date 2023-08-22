"""Load and prep fragpipe tandem mass Tag datasets."""

from dataclasses import dataclass

import pandas as pd
from bystro.proteomics.fragpipe_utils import check_df_cols, prep_annotation_df

ABUNDANCE_COLS = ["Index", "NumberPSM", "ProteinID", "MaxPepProb", "ReferenceIntensity"]


@dataclass(frozen=True)
class TandemMassTagDataset:
    """Represent a Fragpipe Tandem Mass Tag dataset."""

    abundance_df: pd.DataFrame
    annotation_df: pd.DataFrame


def _prep_abundance_df(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """Prep abundance_df, setting index and normalizing abundances by ReferenceIntensity."""
    check_df_cols(abundance_df, ABUNDANCE_COLS)
    abundance_df = abundance_df.set_index("Index")
    first_sample_column_idx = abundance_df.columns.to_list().index("ReferenceIntensity") + 1
    sample_columns = abundance_df.columns[first_sample_column_idx:]
    for sample_column in sample_columns:
        abundance_df[sample_column] -= abundance_df.ReferenceIntensity
    return abundance_df


def load_tandem_mass_tag_dataset(
    abundance_filename: str, annotation_filename: str
) -> TandemMassTagDataset:
    """Load and prep Fragpipe tandem mass tag datasets."""
    raw_abundance_df = pd.read_csv(abundance_filename, sep="\t")
    raw_annotation_df = pd.read_csv(annotation_filename, sep="\t")
    abundance_df = _prep_abundance_df(raw_abundance_df)
    annotation_df = prep_annotation_df(raw_annotation_df)
    return TandemMassTagDataset(abundance_df, annotation_df)
