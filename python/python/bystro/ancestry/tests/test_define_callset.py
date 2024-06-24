"""Test define_callset.py."""

import pandas as pd
import pytest

from bystro.ancestry.define_callset import (
    _get_variants_from_affymetrix_df,
    _get_variants_from_illumina_df,
    liftover_38_from_37,
)

pd.options.future.infer_string = True  # type: ignore

@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("chr22:51156732:G:A", "chr22:50718304:G:A"),
        ("chr12:126890980:G:A", "chr12:126406434:G:A"),
        ("chrX:81328:A:G", "chrX:31328:A:G"),
        ("chr1:900000000:G:A", None),
    ],
)
@pytest.mark.skip(reason="UCSC liftover service may be down.")
def test_liftover_38_from_37(test_input: str, expected: str | None):
    assert expected == liftover_38_from_37(test_input)


@pytest.mark.skip(reason="UCSC liftover service may be down.")
def test__process_affymetrix_df():
    affymetrix_df = pd.DataFrame(
        {
            "Chromosome": {"AFFX-SP-000001": "10", "AFFX-SP-000002": "12", "AFFX-SP-000003": "10"},
            "Physical Position": {
                "AFFX-SP-000001": 123096468,
                "AFFX-SP-000002": 23201352,
                "AFFX-SP-000003": 33545464,
            },
            "Ref Allele": {"AFFX-SP-000001": "C", "AFFX-SP-000002": "C", "AFFX-SP-000003": "G"},
            "Alt Allele": {"AFFX-SP-000001": "G", "AFFX-SP-000002": "G", "AFFX-SP-000003": "C"},
        }
    )
    expected_output = pd.Series(
        {
            0: "chr10:121336954:C:G",
            1: "chr12:23048418:C:G",
            2: "chr10:33256536:G:C",
        }
    )

    assert (expected_output == _get_variants_from_affymetrix_df(affymetrix_df)).all()


@pytest.mark.skip(reason="UCSC liftover service may be down.")
def test__process_illumina_df():
    illumina_df = pd.DataFrame(
        {
            "Chr": {1: "9", 3: "2", 5: "2"},
            "MapInfo": {1: 139926402.0, 3: 220089685.0, 5: 220075045.0},
            "SNP": {1: "[A/G]", 3: "[C/G]", 5: "[T/C]"},
            "RefStrand": {1: "-", 3: "-", 5: "+"},
        }
    )
    expected_output = pd.Series(
        {0: "chr9:137031950:T:C", 1: "chr2:219224963:G:C", 2: "chr2:219210323:T:C"}
    )
    assert (expected_output == _get_variants_from_illumina_df(illumina_df)).all()
