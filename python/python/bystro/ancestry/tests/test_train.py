"""Tests for ancestry model training code."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bystro.ancestry.train import _is_autosomal_variant, _parse_vcf_from_file_stream


def test__is_autosomal_variant():
    assert _is_autosomal_variant("chr1:123456:A:T")
    assert _is_autosomal_variant("chr22:1:T:A")
    assert not _is_autosomal_variant("22:1:A:G")
    assert not _is_autosomal_variant("chrX:1:A:G")
    assert not _is_autosomal_variant("chr23:1:G:C")
    assert not _is_autosomal_variant("chr22:1:A:")
    assert not _is_autosomal_variant("chr22:1:A:AT")
    assert not _is_autosomal_variant("chr22:1:GC:AT")
    assert not _is_autosomal_variant("chr22:1:X:Y")
    with pytest.raises(ValueError, match="cannot have identical ref and alt alleles"):
        _is_autosomal_variant("chr22:1:A:A")


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
    )
    assert_frame_equal(expected_df, actual_df)
