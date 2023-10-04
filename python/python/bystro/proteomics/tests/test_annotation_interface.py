import pytest
from unittest.mock import Mock
from bystro.beanstalkd.worker import ProgressPublisher
from bystro.proteomics.annotation_interface import (
    _preprocess_query,
    _process_response,
    _package_opensearch_query_from_query_string,
    get_samples_and_genes,
    run_annotation_query,
    _process_query_ray,
    _process_query,
)
from bystro.proteomics.tests.test_response import TEST_RESPONSE
from bystro.search.utils.annotation import get_delimiters
from bystro.search.utils.messages import SaveJobData
from bystro.utils.config import get_opensearch_config
import opensearchpy

OPENSEARCH_CONFIG = get_opensearch_config()
import logging

logger = logging.getLogger(__name__)


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
        def __init__(*args, **kwargs):
            logger.info("making mock: %s, %s", args, kwargs)
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


# def test__process_query_inner(monkeypatch):
#     query = {
#         "body": _preprocess_query(
#             _package_opensearch_query_from_query_string(
#                 "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
#             )
#         )
#     }

#     class MockOpenSearchClient:
#         def __init__(*args, **kwargs):
#             pass

#         def search(*args, **kwargs):
#             return TEST_RESPONSE

#     monkeypatch.setattr(opensearchpy, "OpenSearch", MockOpenSearchClient)
#     _process_query_inner(query, OPENSEARCH_CONFIG)
