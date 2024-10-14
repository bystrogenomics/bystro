"""Tests for ancestry model training code."""

import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bystro.ancestry.train import (
    POPS,
    SUPERPOPS,
    _parse_vcf_from_file_stream,
    superpop_predictions_from_pop_probs,
    superpop_probs_from_pop_probs,
)

pd.options.future.infer_string = True  # type: ignore


def test__parse_vcf_from_file_stream():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1 sample2 sample3",
        "chr1	1	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
        "chr1	123	.	T	G	.	PASS	i;n;f;o	GT	0|0	1|1	1|1",
        "chr1	123456	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
    ]
    expected_df = pd.DataFrame(
        [[1.0, 0.0, 1.0], [1.0, 2.0, 1.0], [0.0, 2.0, 0.0]],
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


def test__parse_vcf_from_file_stream_missing_data():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1 sample2 sample3",
        "chr1	1	.	T	G	.	PASS	i;n;f;o	GT	.|.	1|0	0|0",
        "chr1	123	.	T	G	.	PASS	i;n;f;o	GT	0|0	.|1	1|1",
        "chr1	123456	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|.",
    ]
    expected_df = pd.DataFrame(
        [[np.nan, 0.0, 1.0], [1.0, np.nan, 1.0], [0.0, 2.0, np.nan]],
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


def test__parse_vcf_from_file_stream_no_chr_prefix():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1 sample2 sample3",
        "1	1	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
    ]
    expected_df = pd.DataFrame(
        [[1.0], [1.0], [0.0]],
        index=["sample1", "sample2", "sample3"],
        columns=["chr1:1:T:G"],
    )
    actual_df = _parse_vcf_from_file_stream(
        file_stream,
        [
            "chr1:1:T:G",
        ],
        return_exact_variants=True,
    )
    assert_frame_equal(expected_df, actual_df)


def test__parse_vcf_from_file_stream_bad_metadata_fields():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	FILTER	INFO	sample1 sample2 sample3",
        "chr1	1	.	T	G	PASS	i;n;f;o	0|1	1|0	0|0",
        "chr1	123	.	T	G	PASS	i;n;f;o	0|0	1|1	1|1",
        "chr1	123456	.	T	G	PASS	i;n;f;o	0|1	1|0	0|0",
    ]

    expected_err_msg = re.escape(
        "vcf does not contain expected metadata columns.  "
        "Expected: ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'], "
        "got: ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'FILTER', 'INFO', 'sample1', 'sample2'] instead."
    )
    with pytest.raises(ValueError, match=expected_err_msg):
        _parse_vcf_from_file_stream(
            file_stream,
            [
                "chr1:1:T:G",
                "chr1:123:T:G",
                "chr1:123456:T:G",
            ],
            return_exact_variants=True,
        )


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


def test__parse_vcf_from_file_stream_ragged_rows():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1 sample2 sample3",
        "chr1	1	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
        "chr1	123	.	T	G	.	PASS	i;n;f;o	GT	0|0	1|1",
        "chr1	123456	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
    ]

    with pytest.raises(ValueError, match="do all genotype rows have the same number of fields?"):
        _parse_vcf_from_file_stream(
            file_stream,
            [
                "chr1:1:T:G",
                "chr1:123:T:G",
                "chr1:123456:T:G",
            ],
            return_exact_variants=False,
        )


def test__parse_vcf_from_file_stream_bad_filter_values():
    file_stream = [
        "##Some comment",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	sample1 sample2 sample3",
        "chr1	1	.	T	G	.	PASS	i;n;f;o	GT	0|1	1|0	0|0",
        "chr1	123	.	T	G	.	.	i;n;f;o	GT	0|0	1|1     1|1",
        "chr1	123456	.	T	G	.	foo	i;n;f;o	GT	0|1	1|0	0|0",
    ]

    expected_df = pd.DataFrame(
        [[1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
        index=["sample1", "sample2", "sample3"],
        columns=["chr1:1:T:G", "chr1:123:T:G"],
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


def test_superpop_probs_from_pop_probs():
    samples = [f"sample{i}" for i in range(len(POPS))]
    # input array is identity matrix, i.e. one 100% prediction per population
    pop_probs = pd.DataFrame(np.eye(len(POPS)), index=samples, columns=POPS)
    superpop_probs = superpop_probs_from_pop_probs(pop_probs)
    # expected output is matrix mapping each population to its superpop
    expected_superpop_probs = pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=samples,
        columns=SUPERPOPS,
    )
    assert_frame_equal(expected_superpop_probs, superpop_probs)


def test_superpop_predictions_from_pop_probs():
    samples = [f"sample{i}" for i in range(len(POPS))]
    # input array is identity matrix, i.e. one 100% prediction per population
    pop_probs = pd.DataFrame(np.eye(len(POPS)), index=samples, columns=POPS)
    superpop_predictions = superpop_predictions_from_pop_probs(pop_probs)
    expected_superpop_predictions = [
        "AFR",
        "AFR",
        "SAS",
        "EAS",
        "EUR",
        "EAS",
        "EAS",
        "AMR",
        "AFR",
        "EUR",
        "EUR",
        "SAS",
        "AFR",
        "EUR",
        "SAS",
        "EAS",
        "EAS",
        "AFR",
        "AFR",
        "AMR",
        "AMR",
        "SAS",
        "AMR",
        "SAS",
        "EUR",
        "AFR",
    ]
    assert expected_superpop_predictions == superpop_predictions
