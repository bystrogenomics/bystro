import pandas as pd
from io import StringIO
from bystro.proteomics.fragpipe_tandem_mass_tag import parse_tandem_mass_tag_files
from pandas.testing import assert_frame_equal

raw_abundance_df = pd.DataFrame(
    {
        "Index": {0: "A1BG", 1: "A1CF", 2: "A2M"},
        "NumberPSM": {0: 324, 1: 94, 2: 1418},
        "ProteinID": {0: "P04217", 1: "Q9NQ94", 2: "P01023"},
        "MaxPepProb": {0: 1.0, 1: 1.0, 2: 1.0},
        "ReferenceIntensity": {0: 30.04433614619398, 1: 26.115334904878328, 2: 30.80208989046958},
        "CPT0088900003": {0: 30.70998409905496, 1: 26.206662684887046, 2: 31.758776802217614},
        "CPT0079270003": {0: 30.20131916027553, 1: 26.34387337368212, 2: 30.570324340069984},
        "CPT0088920001": {0: 30.004127693933963, 1: 26.75685773043072, 2: 30.897660840389992},
        "CPT0079300001": {0: 30.69208966743661, 1: 25.178399273957886, 2: 31.66196473702596},
        "CPT0088550004": {0: 29.09555725177757, 1: 24.49025631535929, 2: 30.13335176679436},
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


def test_parse_tandem_mass_tag_files():
    abundance_handle = StringIO(raw_abundance_df.to_csv(index=None, sep="\t"))
    annotation_handle = StringIO(raw_annotation_df.to_csv(index=None, sep="\t"))
    tmt_dataset = parse_tandem_mass_tag_files(abundance_handle, annotation_handle)
    expected_abundance_df = pd.DataFrame(
        {
            "NumberPSM": {"A1BG": 324, "A1CF": 94, "A2M": 1418},
            "ProteinID": {"A1BG": "P04217", "A1CF": "Q9NQ94", "A2M": "P01023"},
            "MaxPepProb": {"A1BG": 1.0, "A1CF": 1.0, "A2M": 1.0},
            "ReferenceIntensity": {
                "A1BG": 30.04433614619398,
                "A1CF": 26.115334904878328,
                "A2M": 30.80208989046958,
            },
            "CPT0088900003": {
                "A1BG": 0.6656479528609793,
                "A1CF": 0.09132778000871866,
                "A2M": 0.9566869117480294,
            },
            "CPT0079270003": {
                "A1BG": 0.15698301408155046,
                "A1CF": 0.2285384688037908,
                "A2M": -0.23176555039959723,
            },
            "CPT0088920001": {
                "A1BG": -0.04020845226001768,
                "A1CF": 0.6415228255523928,
                "A2M": 0.09557094992041115,
            },
            "CPT0079300001": {
                "A1BG": 0.6477535212426275,
                "A1CF": -0.9369356309204413,
                "A2M": 0.8598748465563801,
            },
            "CPT0088550004": {
                "A1BG": -0.9487788944164102,
                "A1CF": -1.6250785895190383,
                "A2M": -0.6687381236752223,
            },
        }
    )
    expected_abundance_df.index.name = "Index"
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
    assert_frame_equal(expected_abundance_df, tmt_dataset.abundance_df)
    assert_frame_equal(expected_annotation_df, tmt_dataset.annotation_df)
