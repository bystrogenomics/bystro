import gzip as gz
import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
import ray

from bystro.beanstalkd.worker import get_progress_reporter
from bystro.search.save.handler import _process_query  # Make sure to import your function correctly
from bystro.search.utils.annotation import DelimitersConfig


@pytest.fixture
def dosage_matrix_path():
    # Temporary directory for the test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the path for the dosage matrix file
        dosage_matrix_file_path = os.path.join(temp_dir, "dosage_matrix.feather")

        # 10 irrelevant loci, and 2 that are in the search response
        dosage_df = pd.DataFrame(
            {
                "locus": [f"locus_{i}" for i in range(10)] + ["chr1:100:A:T", "chr2:200:G:C"],
                "sample1": [0 for _ in range(10)] + [0, 2],
                "sample2": [1 for _ in range(10)] + [0, 2],
                "sample3": [2 for _ in range(10)] + [2, 1],
            }
        )

        dosage_df.to_feather(dosage_matrix_file_path)

        yield dosage_matrix_file_path


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
                    "_source": {
                        "discordant": [[[False]]],
                        "chrom": [[["chr1"]]],
                        "pos": [[["100"]]],
                        "inputRef": [[["A"]]],
                        "alt": [[["T"]]],
                    }
                },
                {
                    "_source": {
                        "discordant": [[[False]]],
                        "chrom": [[["chr2"]]],
                        "pos": [[["200"]]],
                        "inputRef": [[["G"]]],
                        "alt": [[["C"]]],
                    }
                },
            ],
        }
    }


@patch("bystro.search.save.handler.OpenSearch")
def test_process_query(
    OpenSearchMock, search_client_args, query_args, mocked_opensearch_response, dosage_matrix_path
):

    instance = OpenSearchMock.return_value
    instance.search.return_value = mocked_opensearch_response

    reporter = get_progress_reporter()
    delimiters = DelimitersConfig()

    with tempfile.NamedTemporaryFile(delete=False) as temp_output_file:
        output_chunk_path = temp_output_file.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_output_file:
        dosage_chunk_path = temp_output_file.name

    result = ray.get(
        _process_query.remote(
            query_args,
            search_client_args,
            ["chrom", "pos", "inputRef", "alt", "discordant"],
            None,
            output_chunk_path,
            reporter,
            delimiters,
            dosage_matrix_path,
            dosage_chunk_path,
        )
    )

    assert result == 2  # 2 rows in the search response

    with gz.open(output_chunk_path) as f:
        output_content = f.read()
        assert output_content == b"chr1\t100\tA\tT\tFalse\nchr2\t200\tG\tC\tFalse\n"

    dosage_df = pd.read_feather(dosage_chunk_path)

    assert set(dosage_df["locus"].values) == set(["chr1:100:A:T", "chr2:200:G:C"])
    assert dosage_df.shape == (2, 4)

    dosage_df = dosage_df.set_index("locus", drop=True)
    assert dosage_df.loc["chr1:100:A:T", "sample1"] == 0
    assert dosage_df.loc["chr1:100:A:T", "sample2"] == 0
    assert dosage_df.loc["chr1:100:A:T", "sample3"] == 2
    assert dosage_df.loc["chr2:200:G:C", "sample1"] == 2
    assert dosage_df.loc["chr2:200:G:C", "sample2"] == 2
    assert dosage_df.loc["chr2:200:G:C", "sample3"] == 1

    os.remove(output_chunk_path)
    os.remove(dosage_chunk_path)
