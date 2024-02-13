import tempfile
import pathlib as path

from bystro.prs.tests.test_prs_dosage import (
    _create_feather_file_with_multiple_batches,
    calculate_prs_in_row_batches,
    calculate_prs_in_pandas,
    calculate_prs_columnar_one_batch,
    calculate_prs_columnar_one_batch_as_numpy_array,
    calculate_prs_columnar_batches,
)

temp_dir = tempfile.TemporaryDirectory()
# Create dataset
file_path, weights = _create_feather_file_with_multiple_batches(path.Path(temp_dir.name), 50)


def test_calculate_prs_pandas(benchmark):
    benchmark(calculate_prs_in_pandas, file_path, weights)


def test_calculate_prs_row_batches(benchmark):
    benchmark(calculate_prs_in_row_batches, file_path, weights)


def test_calculate_prs_columnar_one_batch(benchmark):
    benchmark(calculate_prs_columnar_one_batch, file_path, weights)


def test_calculate_prs_columnar_one_batch_as_numpy_array(benchmark):
    benchmark(calculate_prs_columnar_one_batch_as_numpy_array, file_path, weights)


def test_calculate_prs_columnar_batches(benchmark):
    benchmark(calculate_prs_columnar_batches, file_path, weights)
