from unittest.mock import patch

import numpy as np
import pytest
import ray

from bystro.beanstalkd.worker import get_progress_reporter
from bystro.search.save.handler import AsyncQueryProcessor  # Make sure to import your function correctly


@pytest.fixture
def search_client_args():
    return {"host": "localhost", "port": 9200}


@pytest.fixture
def query_args():
    return {"query": {"match_all": {}}, "size": 10}


@pytest.fixture
def mocked_opensearch_response():
    return {
        "hits": {
            "total": {"value": 2},
            "hits": [
                {
                    "_id": 0,
                    "fields": {"chrom": ["chr1"], "pos": ["100"], "inputRef": ["A"], "alt": ["T"]},
                },
                {
                    "_id": 1,
                    "fields": {"chrom": ["chr2"], "pos": ["200"], "inputRef": ["G"], "alt": ["C"]},
                },
            ],
        }
    }


@patch("bystro.search.save.handler.OpenSearch")
async def test_process_query(OpenSearchMock, search_client_args, query_args, mocked_opensearch_response):

    instance = OpenSearchMock.return_value
    instance.search.return_value = mocked_opensearch_response

    reporter = get_progress_reporter()

    actor = AsyncQueryProcessor.remote(search_client_args, reporter) # type: ignore
    result = ray.get(actor.process_query.remote(query_args))

    assert result is not None

    assert len(result) == 2

    [ids, loci] = result

    assert np.array_equal(ids, np.array([0, 1]))  # 2 document ids

    assert loci == ["chr1:100:A:T", "chr2:200:G:C"]
