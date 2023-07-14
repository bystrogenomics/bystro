"""Tests for ancestry model training code."""
import pandas as pd
from pandas.testing import assert_frame_equal

from bystro.ancestry.train import VARIANT_REGEX, _parse_vcf_from_file_stream


def test_VARIANT_REGEX():
    assert VARIANT_REGEX.match("chr1:123456:A:A")
    assert VARIANT_REGEX.match("chr22:1:A:A")
    assert not VARIANT_REGEX.match("22:1:A:A")
    assert not VARIANT_REGEX.match("chrX:1:A:A")
    assert not VARIANT_REGEX.match("chr23:1:A:A")
    assert not VARIANT_REGEX.match("chr22:1:A:")
    assert not VARIANT_REGEX.match("chr22:1:A:AT")
    assert not VARIANT_REGEX.match("chr22:1:GC:AT")
    assert not VARIANT_REGEX.match("chr22:1:X:Y")


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
    )
    assert_frame_equal(expected_df, actual_df)
