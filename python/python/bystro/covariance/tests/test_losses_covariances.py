import numpy as np
from scipy.linalg import logm  # type: ignore
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
    loss_harmonic,
    loss_logeuclidean,
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

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 1], [1, 3]])
    p = 2
    U, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    B_norm_p_minus_1 = np.dot(U, np.diag(np.power(Sigma, p - 1))) @ Vt
    trace_term = np.trace(B_norm_p_minus_1.T @ (A - B))
    expected = (
        schatten_norm(A, p) ** p - schatten_norm(B, p) ** p - p * trace_term
    )
    assert np.isclose(
        bregman_schatten_p_divergence(A, B, p), expected
    ), "Bregman Schatten p-divergence failed for single pair."

    A_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.5, 0.5])
    expected = np.mean([bregman_schatten_p_divergence(A, B, p) for A in A_list])
    assert np.isclose(
        bregman_schatten_p_divergence(np.array(A_list), B, p, weights), expected
    ), "Bregman Schatten p-divergence failed for list and matrix."

    B_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = np.mean(
        [bregman_schatten_p_divergence(A, B, p) for A, B in zip(A_list, B_list)]
    )
    assert np.isclose(
        bregman_schatten_p_divergence(
            np.array(A_list), np.array(B_list), p, weights
        ),
        expected,
    ), "Bregman Schatten p-divergence failed for two lists."


def test_frobenius_loss():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 2], [2, 2]])
    expected = np.linalg.norm(A - B, "fro")
    assert np.isclose(
        frobenius_loss(A, B), expected
    ), "Frobenius loss calculation failed."

    A_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.5, 0.5])
    expected = np.mean([np.linalg.norm(A - B, "fro") for A in A_list])
    assert np.isclose(
        frobenius_loss(np.array(A_list), B, weights), expected
    ), "Frobenius loss failed for list and matrix."

    B_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = np.mean(
        [np.linalg.norm(A - B, "fro") for A, B in zip(A_list, B_list)]
    )
    assert np.isclose(
        frobenius_loss(np.array(A_list), np.array(B_list), weights), expected
    ), "Frobenius loss failed for two lists."


def test_kl_divergence_gaussian():
    mu0 = np.zeros(2)
    Sigma0 = np.eye(2)
    mu1 = np.zeros(2)
    Sigma1 = np.eye(2)
    expected = 0.  # KL divergence should be 0 if the distributions are the same
    assert np.isclose(
        kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1), expected
    ), "KL divergence for identical distributions failed."

    mu_list = [np.zeros(2), np.ones(2)]
    Sigma_list = [np.eye(2), 2 * np.eye(2)]
    weights = np.array([0.5, 0.5])
    expected = float(np.mean(
        [
            kl_divergence_gaussian(mu, Sigma, mu1, Sigma1)
            for mu, Sigma in zip(mu_list, Sigma_list)
        ]
    ))
    assert np.isclose(
        kl_divergence_gaussian(
            np.array(mu_list), np.array(Sigma_list), mu1, Sigma1, weights
        ),
        expected,
    ), "KL divergence for list against single failed."

    mu1_list = [np.zeros(2), -np.ones(2)]
    Sigma1_list = [np.eye(2), 3 * np.eye(2)]
    expected = float(np.mean(
        [
            kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1)
            for mu0, Sigma0, mu1, Sigma1 in zip(
                mu_list, Sigma_list, mu1_list, Sigma1_list
            )
        ]
    ))
    assert np.isclose(
        kl_divergence_gaussian(
            np.array(mu_list),
            np.array(Sigma_list),
            np.array(mu1_list),
            np.array(Sigma1_list),
            weights,
        ),
        expected,
    ), "KL divergence for two lists failed."


def test_symmetric_kl_divergence_gaussian():
    mu0 = np.zeros(2)
    Sigma0 = np.eye(2)
    mu1 = np.zeros(2)
    Sigma1 = np.eye(2)
    expected = 0.  # Symmetric KL divergence should be 0 if the distributions are the same
    assert np.isclose(
        symmetric_kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1), expected
    ), "Symmetric KL divergence for identical distributions failed."

    mu_list = [np.zeros(2), np.ones(2)]
    Sigma_list = [np.eye(2), 2 * np.eye(2)]
    weights = np.array([0.5, 0.5])
    expected = float(np.mean(
        [
            symmetric_kl_divergence_gaussian(mu, Sigma0, mu1, Sigma1)
            for mu, Sigma0 in zip(mu_list, Sigma_list)
        ]
    ))
    assert np.isclose(
        symmetric_kl_divergence_gaussian(
            np.array(mu_list), np.array(Sigma_list), mu1, Sigma1, weights
        ),
        expected,
    ), "Symmetric KL divergence for list against single failed."

    # Test with two lists of distributions
    mu1_list = [np.zeros(2), -np.ones(2)]
    Sigma1_list = [np.eye(2), 3 * np.eye(2)]
    expected = float(np.mean(
        [
            symmetric_kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1)
            for mu0, Sigma0, mu1, Sigma1 in zip(
                mu_list, Sigma_list, mu1_list, Sigma1_list
            )
        ]
    ))
    assert np.isclose(
        symmetric_kl_divergence_gaussian(
            np.array(mu_list),
            np.array(Sigma_list),
            np.array(mu1_list),
            np.array(Sigma1_list),
            weights,
        ),
        expected,
    ), "Symmetric KL divergence for two lists failed."


def test_mahalanobis_divergence():
    A = np.eye(2)
    B = np.eye(2)
    expected = 0
    assert np.isclose(
        mahalanobis_divergence(A, B), expected
    ), "Mahalanobis divergence calculation failed."

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 0], [0, 1]])
    expected = np.trace(np.dot(A, A.T) - 2 * np.dot(A, B.T) + np.dot(B, B.T))
    assert np.isclose(
        mahalanobis_divergence(A, B), expected
    ), "Mahalanobis divergence calculation failed for single pair."

    A_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.5, 0.5])
    expected = np.mean(
        [
            np.trace(np.dot(A, A.T) - 2 * np.dot(A, B.T) + np.dot(B, B.T))
            for A in A_list
        ]
    )
    assert np.isclose(
        mahalanobis_divergence(np.array(A_list), B, weights), expected
    ), "Mahalanobis divergence failed for list and matrix."

    B_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = np.mean(
        [
            np.trace(np.dot(A, A.T) - 2 * np.dot(A, B.T) + np.dot(B, B.T))
            for A, B in zip(A_list, B_list)
        ]
    )
    assert np.isclose(
        mahalanobis_divergence(np.array(A_list), np.array(B_list), weights),
        expected,
    ), "Mahalanobis divergence failed for two lists."


def test_stein_loss():
    A = np.eye(2)
    B = np.eye(2)
    expected = np.trace(
        A @ np.linalg.inv(B) - np.log(A @ np.linalg.inv(B)) - np.eye(2)
    )
    assert np.isclose(
        stein_loss(A, B), expected
    ), "Stein loss calculation failed."

    S = np.eye(2)
    Sigma = np.eye(2) * 2
    Sigma_inv = np.linalg.inv(Sigma)
    expected = (
        np.trace(S @ Sigma_inv) - np.log(np.linalg.det(S @ Sigma_inv)) - 2
    )
    assert np.isclose(
        stein_loss(S, Sigma), expected
    ), "Stein loss calculation failed for single pair."

    S_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.5, 0.5])
    expected = np.mean(
        [
            np.trace(S @ Sigma_inv) - np.log(np.linalg.det(S @ Sigma_inv)) - 2
            for S in S_list
        ]
    )
    assert np.isclose(
        stein_loss(np.array(S_list), Sigma, weights), expected
    ), "Stein loss failed for list and matrix."

    Sigma_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = np.mean(
        [
            np.trace(S @ np.linalg.inv(Sigma))
            - np.log(np.linalg.det(S @ np.linalg.inv(Sigma)))
            - 2
            for S, Sigma in zip(S_list, Sigma_list)
        ]
    )
    assert np.isclose(
        stein_loss(np.array(S_list), np.array(Sigma_list), weights), expected
    ), "Stein loss failed for two lists."


def test_von_neumann_relative_entropy():
    Sigma = np.eye(2)
    S = np.eye(2)
    expected = 0
    assert np.isclose(
        von_neumann_relative_entropy(Sigma, S), expected
    ), "Von Neumann relative entropy calculation failed."

    Sigma_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.3, 0.7])
    expected = weights.T @ np.array(
        [
            np.trace(Sigma @ logm(Sigma) - Sigma @ logm(S))
            - np.trace(Sigma - S)
            for Sigma in Sigma_list
        ]
    )
    assert np.isclose(
        von_neumann_relative_entropy(np.array(Sigma_list), S, weights), expected
    ), "von Neumann relative entropy failed for list and matrix."

    S_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = weights.T @ np.array(
        [
            np.trace(Sigma @ logm(Sigma) - Sigma @ logm(S))
            - np.trace(Sigma - S)
            for Sigma, S in zip(Sigma_list, S_list)
        ]
    )
    assert np.isclose(
        von_neumann_relative_entropy(
            np.array(Sigma_list), np.array(S_list), weights
        ),
        expected,
    ), "von Neumann relative entropy failed for two lists."


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

    A = np.eye(2)
    B = np.eye(2) * 2
    expected = (
        np.trace(A @ np.linalg.inv(B))
        - np.log(np.linalg.det(A @ np.linalg.inv(B)))
        - A.shape[0]
    )
    assert np.isclose(
        logdet_divergence(A, B), expected
    ), "Log-determinant divergence calculation failed for single pair."

    A_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.5, 0.5])
    expected = np.mean(
        [
            np.trace(A @ np.linalg.inv(B))
            - np.log(np.linalg.det(A @ np.linalg.inv(B)))
            - A.shape[0]
            for A in A_list
        ]
    )
    assert np.isclose(
        logdet_divergence(np.array(A_list), B, weights), expected
    ), "Log-determinant divergence failed for list and matrix."

    B_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = np.mean(
        [
            np.trace(A @ np.linalg.inv(B))
            - np.log(np.linalg.det(A @ np.linalg.inv(B)))
            - A.shape[0]
            for A, B in zip(A_list, B_list)
        ]
    )
    assert np.isclose(
        logdet_divergence(np.array(A_list), np.array(B_list), weights), expected
    ), "Log-determinant divergence failed for two lists."


def test_loss_harmonic():
    A = np.eye(2)
    B = np.eye(2) * 2
    expected = np.linalg.norm(np.linalg.inv(A) - np.linalg.inv(B), ord="fro")
    assert np.isclose(
        loss_harmonic(A, B), expected
    ), "Harmonic loss calculation failed for single pair."

    A_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.5, 0.5])
    expected = np.mean(
        [
            np.linalg.norm(np.linalg.inv(A) - np.linalg.inv(B), ord="fro")
            for A in A_list
        ]
    )
    assert np.isclose(
        loss_harmonic(np.array(A_list), B, weights), expected
    ), "Harmonic loss failed for list and matrix."

    B_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = np.mean(
        [
            np.linalg.norm(np.linalg.inv(A) - np.linalg.inv(B), ord="fro")
            for A, B in zip(A_list, B_list)
        ]
    )
    assert np.isclose(
        loss_harmonic(np.array(A_list), np.array(B_list), weights), expected
    ), "Harmonic loss failed for two lists."


def test_loss_logeuclidean():
    A = np.eye(2)
    B = np.eye(2) * 2
    expected = np.linalg.norm(logm(A) - logm(B), ord="fro")
    assert np.isclose(
        loss_logeuclidean(A, B), expected
    ), "Log-Euclidean loss calculation failed for single pair."

    A_list = [np.eye(2), np.eye(2) * 3]
    weights = np.array([0.5, 0.5])
    expected = np.mean(
        [np.linalg.norm(logm(A) - logm(B), ord="fro") for A in A_list]
    )
    assert np.isclose(
        loss_logeuclidean(np.array(A_list), B, weights), expected
    ), "Log-Euclidean loss failed for list and matrix."

    B_list = [np.eye(2) * 2, np.eye(2) * 3]
    expected = np.mean(
        [
            np.linalg.norm(logm(A) - logm(B), ord="fro")
            for A, B in zip(A_list, B_list)
        ]
    )
    assert np.isclose(
        loss_logeuclidean(np.array(A_list), np.array(B_list), weights), expected
    ), "Log-Euclidean loss failed for two lists."
