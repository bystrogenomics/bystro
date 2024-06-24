"""Load and prep fragpipe tandem mass Tag datasets."""

from io import StringIO

from msgspec import Struct
import pandas as pd

from bystro.proteomics.fragpipe_utils import check_df_starts_with_cols, prep_annotation_df

pd.options.future.infer_string = True  # type: ignore

FRAGPIPE_ABUNDANCE_COLS = ["Index", "NumberPSM", "ProteinID", "MaxPepProb", "ReferenceIntensity"]
FRAGPIPE_GENE_NAME_COLUMN_ORIGINAL = "Index"
FRAGPIPE_GENE_GENE_NAME_COLUMN_RENAMED = "gene_name"
FRAGPIPE_GENE_COLUMN_MAPPING = {
    FRAGPIPE_GENE_NAME_COLUMN_ORIGINAL: FRAGPIPE_GENE_GENE_NAME_COLUMN_RENAMED
}
FRAGPIPE_RENAMED_COLUMNS = list(
    map(lambda x: FRAGPIPE_GENE_COLUMN_MAPPING.get(x, x), FRAGPIPE_ABUNDANCE_COLS)
)

FRAGPIPE_SAMPLE_COLUMN = "sample"
FRAGPIPE_SAMPLE_INTENSITY_COLUMN = "normalized_sample_intensity"


class TandemMassTagDataset(Struct, frozen=True):
    """Represent a Fragpipe Tandem Mass Tag dataset."""

    abundance_df: pd.DataFrame
    annotation_df: pd.DataFrame

    def __post_init__(self) -> None:
        try:
            check_df_starts_with_cols(self.abundance_df, FRAGPIPE_RENAMED_COLUMNS)
        except ValueError as e:
            err_msg = "Received abundance_df with unexpected columns"
            raise ValueError(err_msg) from e

    def get_melted_abundance_df(self) -> pd.DataFrame:
        """Return a melted abundance df with columns [gene_name, sample_id, value]"""
        abundance_df = self.abundance_df

        long_format_df = abundance_df.melt(
            id_vars=FRAGPIPE_RENAMED_COLUMNS,
            var_name=FRAGPIPE_SAMPLE_COLUMN,
            value_name=FRAGPIPE_SAMPLE_INTENSITY_COLUMN,
        )

        return long_format_df


def _prep_abundance_df(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """Prep abundance_df, setting index and normalizing abundances by ReferenceIntensity."""
    check_df_starts_with_cols(abundance_df, FRAGPIPE_ABUNDANCE_COLS)
    abundance_df = abundance_df.rename(columns=FRAGPIPE_GENE_COLUMN_MAPPING)

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
