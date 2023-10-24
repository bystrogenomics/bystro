from pathlib import Path
import re
from typing import Any

import msgspec
import numpy as np
import pandas as pd
import pytest

from bystro.proteomics.annotation_interface import (
    _process_response,
    join_annotation_results_to_proteomics,
    get_samples_and_genes_from_query,
)

TEST_RESPONSE_FILENAME = Path(__file__).parent / "test_response.dat"

with TEST_RESPONSE_FILENAME.open("rb") as f:
    TEST_RESPONSE = msgspec.msgpack.decode(f.read())  # noqa: S301 (data is safe)


class MockOpenSearch:
    def __init__(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        pass

    def search(*args, **kwargs) -> dict:
        del args, kwargs
        return TEST_RESPONSE

    def count(*args, **kwargs) -> dict:
        del args, kwargs
        return {"count": 1}

    def create_point_in_time(*args, **kwargs) -> dict:
        del args, kwargs
        return {"pit_id": 12345}

    def delete_point_in_time(*args, **kwargs) -> None:
        del args, kwargs
        return


def test_get_samples_and_genes_from_query():
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

    mock_client = MockOpenSearch()
    samples_and_genes_df = get_samples_and_genes_from_query(user_query_string, index_name, mock_client)
    assert (1610, 7) == samples_and_genes_df.shape


def tests__process_response():
    ans = _process_response(TEST_RESPONSE)
    assert (1610, 7) == ans.shape
    assert {"1805", "1847", "4805"} == set(ans.sample_id.unique())
    assert 689 == len(ans.gene_name.unique())
    # it's awkward to test for equality of NaN objects, so fill them
    # and compare the filled sets instead.
    MISSING_GENO_VALUE = -1
    expected_dosage_values = {1.0, 2.0, MISSING_GENO_VALUE}
    actual_dosage_values = set(ans.dosage.fillna(MISSING_GENO_VALUE).unique())
    assert expected_dosage_values == actual_dosage_values


def test_join_annotation_results_to_proteomics():
    rng = np.random.default_rng(1337)

    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = None
    mock_client = MockOpenSearch()
    query_result_df = get_samples_and_genes_from_query(user_query_string, index_name, mock_client)

    shared_proteomics_sample_ids = sorted(set(query_result_df.sample_id))[:2]
    all_proteomics_sample_ids = shared_proteomics_sample_ids + [f"sample_id{i}" for i in range(10)]

    shared_proteomics_gene_names = sorted(set(query_result_df.gene_name))[:100]
    all_proteomics_gene_names = shared_proteomics_gene_names + [f"gene_name{i}" for i in range(10)]

    proteomics_df = pd.DataFrame(
        rng.random((len(all_proteomics_sample_ids), len(all_proteomics_gene_names))),
        columns=all_proteomics_gene_names,
        index=all_proteomics_sample_ids,
    )
    joined_df = join_annotation_results_to_proteomics(query_result_df, proteomics_df)
    assert (140, 8) == joined_df.shape
    assert set(shared_proteomics_sample_ids) == set(joined_df.sample_id)
    assert set(shared_proteomics_gene_names) == set(joined_df.gene_name)


def test_join_annotation_results_to_proteomics_bad_query_result_df():
    query_result_df = pd.DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 4, "B": 5, "C": 6}])
    proteomics_df = pd.DataFrame([{"sample_id": "foo", "gene1": 0.5, "gene2": 1}])
    err_msg = (
        "Expected query_result columns (Index(['A', 'B', 'C'], dtype='object')) "
        "to equal: ['sample_id', 'chrom', 'pos', 'ref', 'alt', 'gene_name', 'dosage']"
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        _joined_df = join_annotation_results_to_proteomics(query_result_df, proteomics_df)
