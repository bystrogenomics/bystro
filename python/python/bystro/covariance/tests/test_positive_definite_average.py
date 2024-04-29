import pytest
import numpy as np
from scipy.linalg import logm, expm # type: ignore
from bystro.covariance.positive_definite_average import (
    is_positive_definite,
    pd_mean_harmonic,
    pd_mean_karcher,
    pd_mean_log_euclidean,
)


def test_is_positive_definite():
    A = np.array([[2, 0], [0, 3]])
    B = np.array([[1, 2], [2, 1]])  # Not positive definite
    assert is_positive_definite(A) is True
    assert is_positive_definite(B) is False


def test_pd_mean_harmonic():
    A = np.array([[2, 0], [0, 3]])
    B = np.array([[4, 0], [0, 5]])
    result = pd_mean_harmonic([A, B])
    expected = np.array([[2.66667, 0], [0, 3.75]])
    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    with pytest.raises(ValueError):
        pd_mean_harmonic([])  # Empty list
    with pytest.raises(ValueError):
        pd_mean_harmonic(
            [A, np.array([[1, 2], [2, 1]])]
        )  # Not all positive definite


def test_pd_mean_karcher():
    A = np.array([[3, 0], [0, 3]])
    B = np.array([[2, 0], [0, 4]])
    result = pd_mean_karcher([A, B])
    expected = np.array([[2.5, 0], [0, 3.5]])
    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    with pytest.raises(ValueError):
        pd_mean_karcher([])  # Empty list


def test_log_euclidean_mean_valid_input():
    A = np.array([[2, 0], [0, 3]])
    B = np.array([[1, 0], [0, 1]])
    C = np.array([[4, 0], [0, 5]])
    matrices = [A, B, C]
    expected_result = expm((logm(A) + logm(B) + logm(C)) / 3)
    result = pd_mean_log_euclidean(matrices)
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_log_euclidean_mean_empty_list():
    with pytest.raises(
        ValueError,
        match="All matrices must be positive definite and non-empty.",
    ):
        pd_mean_log_euclidean([])
