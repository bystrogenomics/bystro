import opensearchpy
import pytest
from bystro.proteomics.annotation_interface import (
    _process_response,
    get_samples_and_genes,
)
from bystro.proteomics.tests.sample_response_data import TEST_RESPONSE
from bystro.utils.config import get_opensearch_config

OPENSEARCH_CONFIG = get_opensearch_config()


@pytest.mark.network()
def test_get_samples_and_genes():
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    SAMPLE_INDEX_NAME = "64c889415acb6d3b3e40e07b_63ddc9ce1e740e0020c39928"
    samples_and_genes_df = get_samples_and_genes(user_query_string, SAMPLE_INDEX_NAME)
    assert 1191 == len(samples_and_genes_df)


def test_get_samples_and_genes_unit(monkeypatch):
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

    class MockOpenSearch:
        def __init__(*args, **kwargs) -> None:
            pass

        def search(*args, **kwargs):
            return TEST_RESPONSE

        def count(*args, **kwargs):
            return {"count": 1}

        def create_point_in_time(*args, **kwargs):
            return {"pit_id": 12345}

        def delete_point_in_time(*args, **kwargs):
            return None

    monkeypatch.setattr(opensearchpy.OpenSearch, "__new__", MockOpenSearch)
    samples_and_genes_df = get_samples_and_genes(user_query_string, index_name)
    assert 1191 == len(samples_and_genes_df)


def tests__process_response():
    ans = _process_response(TEST_RESPONSE)
    assert len(ans) == 1191
