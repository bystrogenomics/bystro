from dataclasses import dataclass
import pandas as pd
from typing import TypeVar, Tuple

ABUNDANCE_COLS = ["Index", "NumberPSM", "ProteinID", "MaxPepProb", "ReferenceIntensity"]
ANNOTATION_COLS = ["plex", "channel", "sample", "sample_name", "condition", "replicate"]

T = TypeVar("T")


@dataclass(frozen=True)
class TMTDataset:
    abundance_df: pd.DataFrame
    annotation_df: pd.DataFrame


def is_prefix(xs: list[T], ys: list[T]) -> bool:
    for x, y in zip(xs, ys, strict=False):
        if x != y:
            return False
    return True


def parse_abundance_df(abundance_df: pd.DataFrame) -> pd.DataFrame:
    if not is_prefix(ABUNDANCE_COLS, abundance_df.columns):
        raise ValueError
    abundance_df = abundance_df.set_index("Index")
    first_sample_column_idx = abundance_df.columns.to_list().index("ReferenceIntensity") + 1
    sample_columns = abundance_df.columns[first_sample_column_idx:]
    for sample_column in sample_columns:
        abundance_df[sample_column] -= abundance_df.ReferenceIntensity
    return abundance_df


def parse_annotation_df(annotation_df: pd.DataFrame) -> pd.DataFrame:
    if not is_prefix(ANNOTATION_COLS, annotation_df.columns):
        raise ValueError

    return annotation_df.set_index("sample")


def parse_tandem_mass_tag_files(
    abundance_filename: str, annotation_filename: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_abundance_df = pd.read_csv(abundance_filename, sep="\t")
    raw_annotation_df = pd.read_csv(annotation_filename, sep="\t")
    abundance_df = parse_abundance_df(raw_abundance_df)
    annotation_df = parse_annotation_df(raw_annotation_df)
    return TMTDataset(abundance_df, annotation_df)
