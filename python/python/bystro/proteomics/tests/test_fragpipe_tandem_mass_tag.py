from io import StringIO

import pandas as pd
from bystro.proteomics.fragpipe_tandem_mass_tag import (
    load_tandem_mass_tag_dataset,
)
from pandas.testing import assert_frame_equal

pd.options.future.infer_string = True  # type: ignore

raw_abundance_df = pd.DataFrame(
    {
        "Index": {0: "A1BG", 1: "A1CF", 2: "A2M"},
        "NumberPSM": {0: 324, 1: 94, 2: 1418},
        "ProteinID": {0: "P04217", 1: "Q9NQ94", 2: "P01023"},
        "MaxPepProb": {0: 1.0, 1: 1.0, 2: 1.0},
        "ReferenceIntensity": {0: 30.0, 1: 26.1, 2: 30.8},
        "CPT0088900003": {0: 30.7, 1: 26.2, 2: 31.7},
        "CPT0079270003": {0: 30.2, 1: 26.3, 2: 30.5},
        "CPT0088920001": {0: 30.0, 1: 26.7, 2: 30.8},
        "CPT0079300001": {0: 30.6, 1: 25.1, 2: 31.6},
        "CPT0088550004": {0: 29.0, 1: 24.4, 2: 30.1},
    }
)

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

expected_abundance_df = pd.DataFrame(
    {
        "NumberPSM": {"A1BG": 324, "A1CF": 94, "A2M": 1418},
        "ProteinID": {"A1BG": "P04217", "A1CF": "Q9NQ94", "A2M": "P01023"},
        "MaxPepProb": {"A1BG": 1.0, "A1CF": 1.0, "A2M": 1.0},
        "ReferenceIntensity": {
            "A1BG": 30.0,
            "A1CF": 26.1,
            "A2M": 30.8,
        },
        "CPT0088900003": {
            "A1BG": 0.7,
            "A1CF": 0.1,
            "A2M": 0.9,
        },
        "CPT0079270003": {
            "A1BG": 0.2,
            "A1CF": 0.2,
            "A2M": -0.3,
        },
        "CPT0088920001": {
            "A1BG": -0.0,
            "A1CF": 0.6,
            "A2M": 0.0,
        },
        "CPT0079300001": {
            "A1BG": 0.6,
            "A1CF": -1.0,
            "A2M": 0.8,
        },
        "CPT0088550004": {
            "A1BG": -1.0,
            "A1CF": -1.7,
            "A2M": -0.7,
        },
    }
)
expected_abundance_df.index.name = "Index"
expected_abundance_df = expected_abundance_df.reset_index().rename(columns={"Index": "gene_name"})

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


def test_load_tandem_mass_tag_dataset():
    abundance_handle = StringIO(raw_abundance_df.to_csv(index=False, sep="\t"))
    annotation_handle = StringIO(raw_annotation_df.to_csv(index=False, sep="\t"))
    tandem_mass_tag_dataset = load_tandem_mass_tag_dataset(abundance_handle, annotation_handle)

    assert_frame_equal(expected_abundance_df, tandem_mass_tag_dataset.abundance_df)
    assert_frame_equal(expected_annotation_df, tandem_mass_tag_dataset.annotation_df)
