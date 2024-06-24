"""Tests for proteomics module."""

import pytest
from bystro.proteomics.proteomics import load_fragpipe_dataset
import io
import pandas as pd

pd.options.future.infer_string = True  # type: ignore
# file contents adapted from Supplementary_Data_Phosphoproteome_DIA,
# Supplementary_Data_Proteome_DIA folders of
# wingolab-bystro-matrixData-opensearch repo

test_csv_contents_type1 = """"Index"\t"Gene"\t"Peptide"\t"ReferenceIntensity"\t"Sample1"\t"Sample2"
"NP_000009.1_52_61"\t"ACADVL"\t"SDSHPSDALTR"\t1.1\t1.2\t1.3
"NP_000010.1_195_200"\t"ACAT1"\t"IHMGSCAENTAK"\t2.1\t2.2\t2.3
"NP_000013.2_155_161"\t"ACVRL1"\t"GLHSELGESSLILK"\t3.1\t3.2\t3.3
"""

test_csv_contents_type2 = """"Index"\t"NumberPSM"\t"Proteins"\t"ReferenceIntensity"\t"Sample1"\t"Sample2"
"A1CF"\t28\t"NP_001185747.1;"\t1.1\t1.2\t1.3
"AAAS"\t69\t"NP_001166937.1;"\t2.1\t2.2\t2.3
"AAED1"\t29\t"NP_714542.1;"\t3.1\t3.2\t3.3
"""


test_csv_contents_bad_format = """"Index"\t"Gene"\t"Peptide"\t"BAD_COL_NAME"\t"Sample1"\t"Sample2"
"NP_000009.1_52_61"\t"ACADVL"\t"SDSHPSDALTR"\t1.1\t1.2\t1.3
"NP_000010.1_195_200"\t"ACAT1"\t"IHMGSCAENTAK"\t2.1\t2.2\t2.3
"NP_000013.2_155_161"\t"ACVRL1"\t"GLHSELGESSLILK"\t3.1\t3.2\t3.3
"""


def test_load_fragpipe_dataset_type1():
    stream = io.StringIO(test_csv_contents_type1)
    fragpipe_df = load_fragpipe_dataset(stream)
    expected_df = pd.DataFrame(
        {
            "ACADVL": {"Sample1": 0.1, "Sample2": 0.2},
            "ACAT1": {"Sample1": 0.1, "Sample2": 0.2},
            "ACVRL1": {"Sample1": 0.1, "Sample2": 0.2},
        }
    ).rename_axis(index="sample", columns="gene")

    pd.testing.assert_frame_equal(expected_df, fragpipe_df)


def test_load_fragpipe_dataset_type2():
    stream = io.StringIO(test_csv_contents_type2)
    fragpipe_df = load_fragpipe_dataset(stream)

    expected_df = pd.DataFrame(
        {
            "A1CF": {"Sample1": 0.1, "Sample2": 0.2},
            "AAAS": {"Sample1": 0.1, "Sample2": 0.2},
            "AAED1": {"Sample1": 0.1, "Sample2": 0.2},
        }
    ).rename_axis(index="sample", columns="gene")

    pd.testing.assert_frame_equal(expected_df, fragpipe_df)


def test_load_fragpipe_dataset_bad_format():
    stream = io.StringIO(test_csv_contents_bad_format)
    with pytest.raises(ValueError, match="Dataset format not recognized"):
        load_fragpipe_dataset(stream)
