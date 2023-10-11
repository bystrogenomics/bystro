from pathlib import Path
from typing import Any

import msgspec
from bystro.proteomics.annotation_interface import (
    _process_response,
    get_samples_and_genes,
)

TEST_RESPONSE_FILENAME = Path(__file__).parent / "test_response.dat"

with TEST_RESPONSE_FILENAME.open("rb") as f:
    TEST_RESPONSE = msgspec.msgpack.decode(f.read())  # noqa: S301 (data is safe)


# @pytest.mark.skip()
def test_get_samples_and_genes_unit(monkeypatch):
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
    samples_and_genes_df = get_samples_and_genes(mock_client, user_query_string, index_name)
    monkeypatch.undo()
    assert 1191 == len(samples_and_genes_df)


def tests__process_response():
    ans = _process_response(TEST_RESPONSE)
    assert len(ans) == 1191
    assert {"1805", "1847", "4805"} == set(ans.sample_id.unique())
    assert 689 == len(ans.gene_name.unique())
    assert {1, 2} == set(ans.dosage.unique())
