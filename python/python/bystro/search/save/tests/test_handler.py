from io import BytesIO
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import ray

from bystro.beanstalkd.worker import get_progress_reporter
from bystro.search.save.handler import (
    AsyncQueryProcessor,
    sort_loci_and_doc_ids,
    filter_annotation,
    filter_dosage_matrix,
)


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

    actor = AsyncQueryProcessor.remote(search_client_args, reporter)  # type: ignore
    result = ray.get(actor.process_query.remote(query_args))

    assert result is not None

    assert len(result) == 2

    [ids, loci] = result

    assert np.array_equal(ids, np.array([0, 1]))  # 2 document ids

    assert loci == ["chr1:100:A:T", "chr2:200:G:C"]


def test_empty_input():
    assert sort_loci_and_doc_ids([]) == ([], 0)


def test_all_none_values():
    assert sort_loci_and_doc_ids([None, None, None]) == ([], 0)


def test_mixed_none_and_non_none_values():
    input_data = [None, [(1, "doc1"), (2, "doc2")], None, [(3, "doc3")]]
    expected_output = ([(1, "doc1"), (2, "doc2"), (3, "doc3")], 3)
    assert sort_loci_and_doc_ids(input_data) == expected_output


def test_single_element():
    assert sort_loci_and_doc_ids([[(1, "doc1")]]) == ([(1, "doc1")], 1)


def test_multiple_elements():
    input_data: list[list[tuple[int, str]] | None] = [[(2, "doc2")], [(1, "doc1"), (3, "doc3")]]
    expected_output = ([(1, "doc1"), (2, "doc2"), (3, "doc3")], 3)
    assert sort_loci_and_doc_ids(input_data) == expected_output


def test_sorting_order():
    input_data: list[list[tuple[int, str]] | None] = [
        [(2, "doc2"), (1, "doc1")],
        [(4, "doc4"), (3, "doc3")],
    ]
    expected_output = ([(1, "doc1"), (2, "doc2"), (3, "doc3"), (4, "doc4")], 4)
    assert sort_loci_and_doc_ids(input_data) == expected_output


def test_counting_hits():
    input_data = [[(1, "doc1")], [(2, "doc2"), (3, "doc3")], None]
    expected_output = ([(1, "doc1"), (2, "doc2"), (3, "doc3")], 3)
    assert sort_loci_and_doc_ids(input_data) == expected_output


def mock_subprocess_popen():
    """Mock subprocess.Popen used in filter_annotation."""
    process_mock = MagicMock()
    # Configure the mock to act as a context manager
    process_mock.__enter__.return_value = process_mock
    process_mock.__exit__.return_value = None

    # Mock the stdout attribute with sample data lines as they would be read from a file
    process_mock.stdout = BytesIO(b"header1\theader2\nrow1\tdata\nrow2\tdata\n")
    # You might also need to mock stdin and other attributes if they are used
    process_mock.stdin = MagicMock()
    return process_mock


@pytest.fixture
def mock_dependencies(mocker):
    """Set up and tear down for the mocked subprocess and other dependencies."""
    # Set up the mock for subprocess.Popen
    popen_mock = MagicMock()
    popen_mock.__enter__.return_value = popen_mock
    popen_mock.__exit__.return_value = None
    popen_mock.stdout = BytesIO(b"header1\theader2\nrow1\tdata\nrow2\tdata\n")
    popen_mock.stdin = MagicMock()
    mocker.patch("subprocess.Popen", return_value=popen_mock)

    stats = MagicMock()
    stats.stdin_cli_stats_command = "mock_command"
    job_data = MagicMock()
    job_data.pipeline = [MagicMock()]
    reporter = MagicMock()
    return stats, job_data, reporter


def test_filter_annotation_excludes_correct_loci(mock_dependencies, mocker):
    stats, job_data, reporter = mock_dependencies
    doc_ids_sorted = [(0, "locus1"), (1, "locus3")]  # Indices of rows to include if not filtered
    n_hits = 2

    # Mock filter function to exclude row2
    filter_fn = mocker.Mock(return_value=False)  # Does not filter anything
    job_data.pipeline[0].make_filter.return_value = filter_fn

    # Call the function
    filtered_loci = filter_annotation(
        stats, "path.gz", "parent_path", job_data, doc_ids_sorted, n_hits, reporter, 1
    )

    # Assertions
    assert filtered_loci == ["locus1", "locus3"]
    filter_fn.assert_called()  # Ensure the filter function was called


def test_filter_annotation_excludes_correct_loci2(mock_dependencies):
    stats, job_data, reporter = mock_dependencies
    doc_ids_sorted = [(0, "locus1"), (1, "locus3")]  # Indices of rows to include if not filtered
    n_hits = 2

    def filter_fn(row: list[bytes]) -> bool:
        """Filter out all rows."""
        return row[0].startswith(b"row2")  # Filters out row2 # noqa: E731

    # To simulate a row being filtered, adjust the filter function and re-run
    job_data.pipeline[0].make_filter.return_value = filter_fn
    filtered_loci = filter_annotation(
        stats, "path.gz", "parent_path", job_data, doc_ids_sorted, n_hits, reporter, 1
    )
    assert filtered_loci == ["locus1"]  # we filtered out the 2nd locus


def test_filter_annotation_excludes_correct_loci3(mock_dependencies):
    stats, job_data, reporter = mock_dependencies
    doc_ids_sorted = [(0, "locus1"), (1, "locus3")]  # Indices of rows to include if not filtered
    n_hits = 2

    def filter_fn(_row: list[bytes]) -> bool:
        """Filter out all rows."""
        return True

    job_data.pipeline[0].make_filter.return_value = filter_fn
    filtered_loci = filter_annotation(
        stats, "path.gz", "parent_path", job_data, doc_ids_sorted, n_hits, reporter, 1
    )
    assert filtered_loci == []  # Now expects an empty list because all are filtered


# Prepare common fixtures
@pytest.fixture
def mock_job_data():
    job_data = MagicMock()
    job_data.submission_id = 123
    return job_data


@pytest.fixture
def common_args(mock_job_data):
    return {
        "dosage_out_path": "mock_dosage_out_path",
        "parent_dosage_matrix_path": "mock_parent_dosage_path",
        "filtered_loci": ["locus1", "locus2"],
        "job_data": mock_job_data,
        "queue_config_path": "mock_queue_config_path",
        "reporting_interval": 10,
        "output_dir": "mock_output_dir",
        "basename": "mock_basename",
    }


@patch("bystro.search.save.handler.logger")
def test_no_dosage_matrix(mock_logger, common_args, mock_dependencies):
    _, _, reporter = mock_dependencies

    with patch("pathlib.Path.touch") as mock_touch:
        filter_dosage_matrix(**common_args, reporter=reporter)
        mock_touch.assert_called_once_with()  # Ensure touch is called to create the file
        mock_logger.info.assert_called_with("No dosage matrix to filter")


@patch("bystro.search.save.handler.logger")
@patch("os.stat")
def test_empty_dosage_matrix(mock_stat, mock_logger, common_args, mock_dependencies):
    _, _, reporter = mock_dependencies
    mock_stat.return_value.st_size = 0
    with patch("pathlib.Path.touch") as mock_touch:
        filter_dosage_matrix(**common_args, reporter=reporter)
        mock_touch.assert_called_once_with()
        mock_logger.info.assert_called_with("No dosage matrix to filter")


@patch("bystro.search.save.handler.logger")
@patch("os.stat")
@patch("builtins.open", new_callable=MagicMock)
def test_normal_operation(mock_open, mock_stat, mock_logger, common_args, mock_dependencies):
    _, _, reporter = mock_dependencies
    mock_stat.return_value.st_size = 1024  # Non-zero size
    with (
        patch("pathlib.Path.touch") as mock_touch,
        patch("bystro.search.save.handler.run_dosage_filter") as mock_run,
        patch("bystro.search.save.handler.Timer") as mock_timer,
    ):
        mock_timer.return_value.__enter__.return_value.elapsed_time = 5
        filter_dosage_matrix(**common_args, reporter=reporter)
        mock_run.assert_called_once()  # Check if the filtering function was indeed called
        mock_touch.assert_not_called()  # Should not touch since matrix exists
        assert mock_open.called
        mock_logger.info.assert_called_with("Filtering dosage matrix took %s seconds", 5)


@patch("os.stat")
@patch("builtins.open", new_callable=MagicMock)
def test_no_loci_provided(mock_open, mock_stat, common_args, mock_dependencies):
    _, _, reporter = mock_dependencies

    # Setting the mock to return a non-zero file size
    mock_stat.return_value.st_size = 1024

    with (
        patch("builtins.open", mock_open()) as mocked_file,
        patch("bystro.search.save.handler.run_dosage_filter") as mock_run_dosage_filter,
    ):
        handle = mocked_file.return_value.__enter__.return_value

        filter_dosage_matrix(**common_args, reporter=reporter)
        mocked_file.assert_called_once_with(
            os.path.join(common_args["output_dir"], "mock_basename_loci.txt"), "w"
        )

        handle.write.assert_called()  # Expect no writes if no loci

        assert handle.write.call_count == 2

        mock_run_dosage_filter.assert_called()  # Ensure the dosage filter function was called
