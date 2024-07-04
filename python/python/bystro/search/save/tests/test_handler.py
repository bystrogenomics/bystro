from io import BytesIO
from unittest.mock import patch, MagicMock

import numpy as np
from numpy.typing import NDArray
import pytest
import ray

from bystro.beanstalkd.messages import get_progress_reporter
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
    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids([])
    assert np.array_equal(doc_ids_sorted, np.array([], dtype=np.int32))
    assert np.array_equal(loci_sorted, np.array([], dtype=object))
    assert n_hits == 0


def test_all_none_values():
    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids(
        [
            (np.array([], dtype=np.int32), np.array([], dtype=object)),
            (np.array([], dtype=np.int32), np.array([], dtype=object)),
            (np.array([], dtype=np.int32), np.array([], dtype=object)),
        ]
    )
    assert np.array_equal(doc_ids_sorted, np.array([], dtype=np.int32))
    assert np.array_equal(loci_sorted, np.array([], dtype=object))
    assert n_hits == 0


def test_mixed_none_and_non_none_values():
    input_data = [
        (np.array([], dtype=np.int32), np.array([], dtype=object)),
        (np.array([1, 2], dtype=np.int32), np.array(["doc1", "doc2"], dtype=object)),
        (np.array([], dtype=np.int32), np.array([], dtype=object)),
        (np.array([3], dtype=np.int32), np.array(["doc3"], dtype=object)),
    ]
    expected_output = (
        np.array([1, 2, 3], dtype=np.int32),
        np.array(["doc1", "doc2", "doc3"], dtype=object),
        3,
    )

    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids(input_data)
    assert np.array_equal(doc_ids_sorted, expected_output[0])
    assert np.array_equal(loci_sorted, expected_output[1])
    assert n_hits == expected_output[2]


def test_single_element():
    input_data: list[tuple[NDArray[np.int32], NDArray]] = [
        (np.array([1], dtype=np.int32), np.array(["doc1"], dtype=object))
    ]
    expected_output = (np.array([1], dtype=np.int32), np.array(["doc1"], dtype=object), 1)
    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids(input_data)
    assert np.array_equal(doc_ids_sorted, expected_output[0])
    assert np.array_equal(loci_sorted, expected_output[1])
    assert n_hits == expected_output[2]


def test_multiple_elements():
    input_data: list[tuple[NDArray[np.int32], NDArray]] = [
        (np.array([2, 1, 3], dtype=np.int32), np.array(["doc2", "doc1", "doc3"], dtype=object))
    ]
    expected_output = (
        np.array([1, 2, 3], dtype=np.int32),
        np.array(["doc1", "doc2", "doc3"], dtype=object),
        3,
    )
    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids(input_data)
    assert np.array_equal(doc_ids_sorted, expected_output[0])
    assert np.array_equal(loci_sorted, expected_output[1])
    assert n_hits == expected_output[2]


def test_sorting_order():
    input_data: list[tuple[NDArray[np.int32], NDArray]] = [
        (np.array([2, 1], dtype=np.int32), np.array(["doc2", "doc1"], dtype=object)),
        (np.array([4, 3], dtype=np.int32), np.array(["doc4", "doc3"], dtype=object)),
    ]
    expected_output = (
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array(["doc1", "doc2", "doc3", "doc4"], dtype=object),
        4,
    )

    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids(input_data)
    assert np.array_equal(doc_ids_sorted, expected_output[0])
    assert np.array_equal(loci_sorted, expected_output[1])
    assert n_hits == expected_output[2]


def test_counting_hits():
    input_data: list[tuple[NDArray[np.int32], NDArray]] = [
        (np.array([1], dtype=np.int32), np.array(["doc1"], dtype=object)),
        (np.array([2, 3], dtype=np.int32), np.array(["doc2", "doc3"], dtype=object)),
        (np.array([], dtype=np.int32), np.array([], dtype=object)),
    ]
    expected_output = (
        np.array([1, 2, 3], dtype=np.int32),
        np.array(["doc1", "doc2", "doc3"], dtype=object),
        3,
    )
    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids(input_data)
    assert np.array_equal(doc_ids_sorted, expected_output[0])
    assert np.array_equal(loci_sorted, expected_output[1])
    assert n_hits == expected_output[2]


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


def test_filter_annotation_excludes_correct_loci(mock_dependencies, mocker, tmp_path):
    stats, job_data, reporter = mock_dependencies
    doc_ids_sorted = np.array([0, 1], dtype=np.int32)
    loci_sorted = np.array(["locus1", "locus3"])
    n_hits = 2
    loci_file_path = tmp_path / "loci.txt"

    # Mock filter function to exclude row2
    filter_fn = mocker.Mock(return_value=False)  # Does not filter anything
    job_data.pipeline[0].make_filter.return_value = filter_fn

    # Call the function
    retained_count = filter_annotation(
        stats,
        "path.gz",
        "parent_path",
        job_data,
        doc_ids_sorted,
        loci_sorted,
        n_hits,
        reporter,
        10,
        loci_file_path,
    )

    # Assertions
    assert retained_count == 2
    filter_fn.assert_called()  # Ensure the filter function was called
    assert loci_file_path.read_text() == "locus1\nlocus3\n"


def test_filter_annotation_excludes_correct_loci2(mock_dependencies, tmp_path):
    stats, job_data, reporter = mock_dependencies
    doc_ids_sorted = np.array([0, 1], dtype=np.int32)
    loci_sorted = np.array(["locus1", "locus3"])
    n_hits = 2
    loci_file_path = tmp_path / "loci.txt"

    # Mock filter function to exclude row2
    def filter_fn(row: list[bytes]) -> bool:
        """Filter out all rows."""
        return row[0].startswith(b"row2")  # Filters out row2 # noqa: E731

    # To simulate a row being filtered, adjust the filter function and re-run
    job_data.pipeline[0].make_filter.return_value = filter_fn

    # Call the function
    retained_count = filter_annotation(
        stats,
        "path.gz",
        "parent_path",
        job_data,
        doc_ids_sorted,
        loci_sorted,
        n_hits,
        reporter,
        10,
        loci_file_path,
    )

    # Assertions
    assert retained_count == 1
    assert loci_file_path.read_text() == "locus1\n"


def test_filter_annotation_excludes_correct_loci3(mock_dependencies, tmp_path):
    stats, job_data, reporter = mock_dependencies
    doc_ids_sorted = np.array([0, 1], dtype=np.int32)
    loci_sorted = np.array(["locus1", "locus3"])
    n_hits = 2
    loci_file_path = tmp_path / "loci.txt"

    # Mock filter function to exclude row2
    def filter_fn(_: list[bytes]) -> bool:
        """Filter out all rows."""
        return True  # filter out all rows

    # To simulate a row being filtered, adjust the filter function and re-run
    job_data.pipeline[0].make_filter.return_value = filter_fn

    # Call the function
    retained_count = filter_annotation(
        stats,
        "path.gz",
        "parent_path",
        job_data,
        doc_ids_sorted,
        loci_sorted,
        n_hits,
        reporter,
        10,
        loci_file_path,
    )

    # Assertions
    assert retained_count == 0
    assert loci_file_path.read_text() == ""


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
        "loci_file_path": "loci.txt",
        "job_data": mock_job_data,
        "queue_config_path": "mock_queue_config_path",
        "reporting_interval": 10,
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


@patch("os.stat")
@patch("os.path.exists")
def test_no_loci_provided(mock_exists, mock_stat, common_args, mock_dependencies):
    _, _, reporter = mock_dependencies

    # Setting the mock to return a non-zero file size for the parent dosage matrix
    mock_stat.side_effect = [
        MagicMock(st_size=1024),
        MagicMock(st_size=0),
    ]  # Non-zero for dosage matrix, zero for loci file
    mock_exists.side_effect = [True, False]  # Exists for dosage matrix, does not exist for loci file

    expected_loci_file_path = common_args["loci_file_path"]

    with (
        patch("bystro.search.save.handler.run_dosage_filter") as mock_run_dosage_filter,
        patch("pathlib.Path.touch") as mock_touch,
    ):
        filter_dosage_matrix(**common_args, reporter=reporter)

        # Assert that the loci file path was checked for existence
        mock_exists.assert_any_call(expected_loci_file_path)

        # Assert no dosage matrix filtering was triggered
        mock_run_dosage_filter.assert_not_called()

        # Assert the log and message functions were called appropriately
        mock_dependencies[2].message.remote.assert_called_once_with("No dosage matrix to filter.")

        mock_touch.assert_called_once_with()
