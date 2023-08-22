import pandas as pd

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
