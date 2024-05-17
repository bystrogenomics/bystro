import numpy as np
from bystro.covariance.losses_covariances import (
    schatten_norm,
    bregman_schatten_p_divergence,
    frobenius_loss,
    kl_divergence_gaussian,
    symmetric_kl_divergence_gaussian,
    mahalanobis_divergence,
    stein_loss,
    von_neumann_relative_entropy,
    logdet_divergence,
)


def test_schatten_norm():
    matrix = np.array([[1, 2], [3, 4]])
    p = 2
    expected = np.power(
        np.sum(np.power(np.linalg.svd(matrix, compute_uv=False), p)), 1 / p
    )
    assert np.isclose(
        schatten_norm(matrix, p), expected
    ), "Schatten norm calculation failed."


def test_bregman_schatten_p_divergence():
    A = np.eye(2)
    B = np.eye(2)
    p = 2
    assert np.isclose(
        bregman_schatten_p_divergence(A, B, p), 0
    ), "Bregman Schatten p divergence calculation failed."


def test_frobenius_loss():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 2], [2, 2]])
    expected = np.linalg.norm(A - B, "fro")
    assert np.isclose(
        frobenius_loss(A, B), expected
    ), "Frobenius loss calculation failed."


def test_kl_divergence_gaussian():
    mu0 = np.zeros(2)
    Sigma0 = np.eye(2)
    mu1 = np.zeros(2)
    Sigma1 = np.eye(2)
    expected = 0  # KL divergence should be 0 if the distributions are the same
    assert np.isclose(
        kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1), expected
    ), "KL divergence for identical distributions failed."


def test_symmetric_kl_divergence_gaussian():
    mu0 = np.zeros(2)
    Sigma0 = np.eye(2)
    mu1 = np.zeros(2)
    Sigma1 = np.eye(2)
    expected = 0  # Symmetric KL divergence should be 0 if the distributions are the same
    assert np.isclose(
        symmetric_kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1), expected
    ), "Symmetric KL divergence for identical distributions failed."


def test_mahalanobis_divergence():
    A = np.eye(2)
    B = np.eye(2)
    expected = 0
    assert np.isclose(
        mahalanobis_divergence(A, B), expected
    ), "Mahalanobis divergence calculation failed."


def test_stein_loss():
    A = np.eye(2)
    B = np.eye(2)
    expected = np.trace(
        A @ np.linalg.inv(B) - np.log(A @ np.linalg.inv(B)) - np.eye(2)
    )
    assert np.isclose(
        stein_loss(A, B), expected
    ), "Stein loss calculation failed."


def test_von_neumann_relative_entropy():
    Sigma = np.eye(2)
    S = np.eye(2)
    expected = 0
    assert np.isclose(
        von_neumann_relative_entropy(Sigma, S), expected
    ), "Von Neumann relative entropy calculation failed."


def test_logdet_divergence():
    A = np.eye(2)
    B = np.eye(2)
    expected = (
        np.trace(A @ np.linalg.inv(B))
        - np.log(np.linalg.det(A @ np.linalg.inv(B)))
        - A.shape[0]
    )
    assert np.isclose(
        logdet_divergence(A, B), expected
    ), "Log-determinant divergence calculation failed."
