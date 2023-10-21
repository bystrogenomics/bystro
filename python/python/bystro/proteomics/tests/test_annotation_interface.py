from pathlib import Path
from typing import Any

import msgspec

from bystro.proteomics.annotation_interface import (
    _process_response,
    get_samples_and_genes_from_query,
)

TEST_RESPONSE_FILENAME = Path(__file__).parent / "test_response.dat"

with TEST_RESPONSE_FILENAME.open("rb") as f:
    TEST_RESPONSE = msgspec.msgpack.decode(f.read())  # noqa: S301 (data is safe)


def test_get_samples_and_genes_from_query():
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

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
