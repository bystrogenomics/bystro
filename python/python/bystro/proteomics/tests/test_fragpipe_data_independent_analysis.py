from io import StringIO
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from bystro.proteomics.fragpipe_data_independent_analysis import load_data_independent_analysis_dataset

pd.options.future.infer_string = True  # type: ignore

raw_annotation_df = pd.DataFrame(
    {
        "plex": {0: 16, 1: 16, 2: 16},
        "channel": {0: "126", 1: "127N", 2: "127C"},
        "sample": {0: "CPT0088900003", 1: "CPT0079270003", 2: "CPT0088920001"},
        "sample_name": {0: "C3N-01179-T", 1: "C3L-00606-T", 2: "C3N-01179-N"},
        "condition": {0: "Tumor", 1: "Tumor", 2: "NAT"},
        "replicate": {0: 1, 1: 1, 2: 1},
    }
)

expected_annotation_df = pd.DataFrame(
    {
        "plex": {"CPT0088900003": 16, "CPT0079270003": 16, "CPT0088920001": 16},
        "channel": {"CPT0088900003": "126", "CPT0079270003": "127N", "CPT0088920001": "127C"},
        "sample_name": {
            "CPT0088900003": "C3N-01179-T",
            "CPT0079270003": "C3L-00606-T",
            "CPT0088920001": "C3N-01179-N",
        },
        "condition": {"CPT0088900003": "Tumor", "CPT0079270003": "Tumor", "CPT0088920001": "NAT"},
        "replicate": {"CPT0088900003": 1, "CPT0079270003": 1, "CPT0088920001": 1},
    }
)
expected_annotation_df.index.name = "sample"


def test_parse_data_independent_analysis_dataset():
    raw_pg_matrix_df = pd.DataFrame(
        {
            "Protein.Group": {0: "A0A024RBG1", 1: "A0A075B6H7", 2: "A0A075B6H9"},
            "Protein.Ids": {0: "A0A024RBG1", 1: "A0A075B6H7", 2: "A0A075B6H9"},
            "Protein.Names": {0: np.nan, 1: np.nan, 2: np.nan},
            "Genes": {0: "NUDT4B", 1: "IGKV3-7", 2: "IGLV4-69"},
            "First.Protein.Description": {0: np.nan, 1: np.nan, 2: np.nan},
            "foo.mzML": {
                0: 806691.0,
                1: 38656400.0,
                2: 129411.0,
            },
            "bar.mzML": {
                0: 1056910.0,
                1: 74868600.0,
                2: np.nan,
            },
            "baz.mzML": {
                0: 1530830.0,
                1: 56854300.0,
                2: np.nan,
            },
            "quux.mzML": {
                0: 1337020.0,
                1: 65506700.0,
                2: 478757.0,
            },
        }
    )

    expected_pg_matrix_df = pd.DataFrame(
        {
            "foo.mzML": {"A0A024RBG1": 806691.0, "A0A075B6H7": 38656400.0, "A0A075B6H9": 129411.0},
            "bar.mzML": {"A0A024RBG1": 1056910.0, "A0A075B6H7": 74868600.0, "A0A075B6H9": np.nan},
            "baz.mzML": {"A0A024RBG1": 1530830.0, "A0A075B6H7": 56854300.0, "A0A075B6H9": np.nan},
            "quux.mzML": {"A0A024RBG1": 1337020.0, "A0A075B6H7": 65506700.0, "A0A075B6H9": 478757.0},
        }
    )
    expected_pg_matrix_df.index.name = "Protein.Ids"

    assert raw_pg_matrix_df is not None
    assert raw_annotation_df is not None

    pg_matrix_handle = StringIO(raw_pg_matrix_df.to_csv(index=False, sep="\t"))
    annotation_handle = StringIO(raw_annotation_df.to_csv(index=False, sep="\t"))
    dia_dataset = load_data_independent_analysis_dataset(pg_matrix_handle, annotation_handle)
    assert_frame_equal(expected_annotation_df, dia_dataset.annotation_df)
    assert_frame_equal(expected_pg_matrix_df, dia_dataset.pg_matrix_df)
