import pytest
import numpy as np
from scipy.linalg import logm, expm  # type: ignore
from bystro.covariance.positive_definite_average import (
    is_positive_definite,
    pd_mean_harmonic,
    pd_mean_karcher,
    pd_mean_log_euclidean,
    median_euclid,
    median_riemann,
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


def test_median_euclid():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(200, 3, 3))
    tol = 1e-5
    maxiter = 50
    init = rng.normal(size=(3, 3))

    result = median_euclid(X, tol=tol, maxiter=maxiter, init=init)
    expected_result = np.array(
        [
            [0.0598997, 0.09976438, 0.12645169],
            [0.05005812, 0.02357827, -0.01803956],
            [0.01140484, -0.04392259, -0.09349429],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)

    rng = np.random.default_rng(2021)
    X = rng.normal(size=(50, 3, 3))
    tol = 1e-4
    maxiter = 50
    init = np.eye(3)

    result = median_euclid(X, tol=tol, maxiter=maxiter, init=init)
    expected_result = np.array(
        [
            [0.0644741, 0.02461982, 0.31533122],
            [-0.09105787, 0.03994839, 0.00068511],
            [-0.04422731, 0.06693973, 0.0914701],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    rng = np.random.default_rng(2021)
    X = rng.normal(size=(5, 3, 3))
    tol = 1e-6
    maxiter = 50
    weights = np.abs(rng.normal(size=(5)))
    init = np.zeros_like(X[0])

    result = median_euclid(
        X, tol=tol, maxiter=maxiter, init=init, weights=weights
    )
    expected_result = np.array(
        [
            [0.14784622, -0.45876689, -0.63059916],
            [0.36933798, 0.2607874, -0.78234202],
            [0.27428875, -1.71872868, 0.15072317],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result, decimal=6)


def test_median_riemann():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(200, 3, 3))
    tol = 1e-5
    maxiter = 50
    init = rng.normal(size=(3, 3))
    init = init.T @ init

    for i in range(len(X)):
        X[i] = X[i].T @ X[i]

    result = median_riemann(X, tol=tol, maxiter=maxiter, init=init)
    expected_result = np.array(
        [
            [0.84792319, -0.06697823, 0.10328062],
            [-0.06697823, 0.84572932, 0.0447306],
            [0.10328062, 0.0447306, 1.14538742],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    rng = np.random.default_rng(2021)
    X = rng.normal(size=(50, 3, 3))
    tol = 1e-4
    maxiter = 50
    init = np.eye(3)

    for i in range(len(X)):
        X[i] = X[i].T @ X[i]

    result = median_riemann(X, tol=tol, maxiter=maxiter, init=init)
    expected_result = np.array(
        [
            [1.02122241, 0.06260142, 0.00918335],
            [0.06260142, 0.79655707, 0.00601193],
            [0.00918335, 0.00601193, 1.19651782],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    rng = np.random.default_rng(2021)
    X = rng.normal(size=(5, 3, 3))

    for i in range(len(X)):
        X[i] = X[i].T @ X[i]
    tol = 1e-6
    maxiter = 50

    result = median_riemann(X, tol=tol, maxiter=maxiter)
    expected_result = np.array(
        [
            [0.77789694, -0.18399038, -0.20124625],
            [-0.18399038, 0.87554047, -0.74517954],
            [-0.20124625, -0.74517954, 2.21098985],
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)
