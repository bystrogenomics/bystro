"""Tests for ancestry model training code."""
import logging

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from bystro.ancestry.train import _parse_vcf_from_file_stream, _parse_vcf_line_for_dosages


def test__parse_vcf_from_file_stream():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1 sample2 sample3",
        "chr1	1	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
        "chr1	123	.	T	G	.	PASS	i;n;f;o	GT	0|0	1|1	1|1",
        "chr1	123456	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
    ]
    expected_df = pd.DataFrame(
        [[1, 0, 1], [1, 2, 1], [0, 2, 0]],
        index=["sample1", "sample2", "sample3"],
        columns=["chr1:1:T:G", "chr1:123:T:G", "chr1:123456:T:G"],
    )
    actual_df = _parse_vcf_from_file_stream(
        file_stream,
        [
            "chr1:1:T:G",
            "chr1:123:T:G",
            "chr1:123456:T:G",
        ],
        return_exact_variants=True,
    )
    assert_frame_equal(expected_df, actual_df)


def test__parse_vcf_from_file_stream_wrong_chromosome():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1 sample2 sample3",
        "chr2	1	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
        "chr2	123	.	T	G	.	PASS	i;n;f;o	GT	0|0	1|1	1|1",
        "chr2	123456	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
    ]
    expected_df = pd.DataFrame(
        [],
        index=["sample1", "sample2", "sample3"],
        columns=[],
    )

    actual_df = _parse_vcf_from_file_stream(
        file_stream,
        [
            "chr1:1:T:G",
            "chr1:123:T:G",
            "chr1:123456:T:G",
        ],
        return_exact_variants=False,
    )
    assert_frame_equal(expected_df, actual_df)

    expected_df_missing_data = pd.DataFrame(
        np.zeros((3, 3)) * np.nan,
        index=["sample1", "sample2", "sample3"],
        columns=[
            "chr1:1:T:G",
            "chr1:123:T:G",
            "chr1:123456:T:G",
        ],
    )

    actual_df_missing_data = _parse_vcf_from_file_stream(
        file_stream,
        [
            "chr1:1:T:G",
            "chr1:123:T:G",
            "chr1:123456:T:G",
        ],
        return_exact_variants=True,
    )
    # check frame equality up to column ordering, which may differ if some variants were missing.
    assert_frame_equal(expected_df_missing_data, actual_df_missing_data, check_like=True)

def test__parse_vcf_line_for_dosages(caplog):
    good_line = "chr1   1   .   T   G   .   PASS    i;n;f;o GT  0|1 1|0 0|0"
    res = _parse_vcf_line_for_dosages(good_line, ["chr1:1:T:G"])
    print("res", res)
    assert res is not None

    variant, dosages = res
    assert variant == "chr1:1:T:G"
    assert dosages == [1, 1, 0]

    res = _parse_vcf_line_for_dosages(good_line, ["chr10:100:A:G"])
    assert res is None

    no_pass_line = "chr1	1	.	T	C	.	.	i;n;f;o	GT	1|1	0|0	0|1"
    res = _parse_vcf_line_for_dosages(no_pass_line, ["chr1:1:T:C"])
    assert res is not None

    variant, dosages = res
    assert variant == "chr1:1:T:C"
    assert dosages == [2, 0, 1]

    non_autosomal = "chrX	1	.	C	T	.	PASS	i;n;f;o	GT	1|1	1|0	0|0"
    res = _parse_vcf_line_for_dosages(non_autosomal, ["chrX:1:C:T"])
    assert res is None

    failing_line = "chr22	2	.	T	G	.	q10	i;n;f;o	GT	0|1	1|0	0|0"
    res = _parse_vcf_line_for_dosages(failing_line, ["chr22:1:T:G"])
    assert res is None

    with caplog.at_level(logging.INFO):
        failing_line2 = "chr1	1	.	T	T	.	PASS	i;n;f;o	GT	0|1	1|0	0|0"
        res = _parse_vcf_line_for_dosages(failing_line2, ["chr1:1:T:A"])
        assert res is None

        assert "Skipping VCF line: Variant chr1:1:T:T cannot have identical ref and alt alleles" in caplog.text