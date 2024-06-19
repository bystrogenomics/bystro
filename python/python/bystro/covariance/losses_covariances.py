"""
This module provides several mathematical functions for computing 
different types of norms and divergences between matrices, useful in 
applications like machine learning, data analysis, and scientific 
computing. It utilizes the numpy and scipy libraries to perform 
efficient numerical operations.

Functions:
- schatten_norm(X, p): Computes the Schatten p-norm of a matrix X.
- bregman_schatten_p_divergence(A, B, p): Computes the Bregman divergence 
  based on the Schatten p-norm between two matrices A and B.
- frobenius_loss(A, B): Calculates the Frobenius loss between two matrices 
  A and B.
- von_neumann_relative_entropy(Sigma, S): Calculates the von Neumann 
  relative entropy between two positive definite matrices.
- logdet_divergence(A, B): Computes the log-determinant divergence between 
  two positive definite matrices.

The mathematical formulations for the Schatten norm and Bregman divergence 
are based on traditional linear algebra definitions. The Frobenius loss, 
von Neumann relative entropy, and log-determinant divergence are 
implemented following principles in numerical analysis and matrix theory.

References:
- "Matrix Analysis", R.A. Horn and C.R. Johnson, Cambridge University Press, 2013.
- "The Matrix Cookbook", Kaare Brandt Petersen and Michael Syskind Pedersen, 2012.
"""

import numpy as np
import numpy.linalg as la
from scipy.linalg import logm  # type: ignore
from typing import Optional


def schatten_norm(X: np.ndarray, p: float) -> float:
    """
    Compute the Schatten p-norm of a matrix X.

    The Schatten p-norm is defined as:
        ||X||_p = (sum(s_i^p))^(1/p)
    where s_i are the singular values of X.

    Parameters
    ----------
    X : np.ndarray
        The input matrix.
    p : float
        The order of the norm.

    Returns
    -------
    float
        The Schatten p-norm of the matrix X.
    """
    singular_values = la.svd(X, compute_uv=False)
    return np.power(np.sum(np.power(singular_values, p)), 1 / p)


def bregman_schatten_p_divergence(
    A: np.ndarray, B: np.ndarray, p: float, weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute the Bregman divergence based on the Schatten p-norm between
    two matrices or sets of matrices A and B.

    The Bregman divergence for Schatten norms is defined as:
        D_p(A, B) = ||A||_p^p - ||B||_p^p - p * tr((B^p-1)^(T) * (A - B))
    where ||.||_p denotes the Schatten p-norm.

    Handles:
    - Single matrices against single matrices.
    - A list of matrices against a single matrix.
    - A list of matrices against a list of matrices of the same length.

    Parameters
    ----------
    A : np.ndarray
        The first input matrix or set of matrices.
    B : np.ndarray
        The second input matrix or set of matrices.
    p : float
        The order of the Schatten norm used in the divergence calculation.
    weights : np.ndarray, optional
        The weights for each matrix, default is uniform weights.

    Returns
    -------
    float
        The average Bregman divergence based on the Schatten p-norm.
    """
    A, B = match_matrices(A, B)
    if weights is None:
        weights = np.ones(A.shape[0])

    divergences = []
    for i in range(A.shape[0]):
        norm_B_p = schatten_norm(B[i], p)
        norm_A_p = schatten_norm(A[i], p)
        U, Sigma, Vt = la.svd(B[i], full_matrices=False)
        B_norm_p_minus_1 = np.dot(U, np.diag(np.power(Sigma, p - 1))) @ Vt
        trace_term = np.trace(B_norm_p_minus_1.T @ (A[i] - B[i]))
        divergence = norm_A_p**p - norm_B_p**p - p * trace_term
        divergences.append(divergence)

    weighted_divergence = np.average(divergences, weights=weights)
    return weighted_divergence


def frobenius_loss(
    A: np.ndarray, B: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the Frobenius loss between two matrices A and B.

    The Frobenius loss is defined as the Frobenius norm of the difference
    between A and B:
        L_F(A, B) = ||A - B||_F
    where ||.||_F denotes the Frobenius norm.

    Parameters
    ----------
    A : np.ndarray
        The first input matrix or set of matrices.
    B : np.ndarray
        The second input matrix or set of matrices, must be of the same shape as A.
    weights : np.ndarray, optional
        The weights for each matrix, default is uniform weights.

    Returns
    -------
    float
        The average Frobenius loss between A and B.
    """
    A, B = match_matrices(A, B)

    if weights is None:
        weights = np.ones(A.shape[0])

    losses = [la.norm(A[i] - B[i], "fro") for i in range(A.shape[0])]
    return np.average(losses, weights=weights)


def kl_divergence_gaussian(
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the Kullback-Leibler divergence between two multivariate
    Gaussian distributions or sets of distributions. This can handle:
    - Single distribution against single distribution.
    - List of distributions against a single distribution.
    - List of distributions against a list of distributions of the same length.

    Parameters
    ----------
    mu0 : np.ndarray
        Mean of the first Gaussian distribution(s).
    Sigma0 : np.ndarray
        Covariance matrix of the first Gaussian distribution(s).
    mu1 : np.ndarray
        Mean of the second Gaussian distribution(s).
    Sigma1 : np.ndarray
        Covariance matrix of the second Gaussian distribution(s).
    weights : np.ndarray, optional
        Weights for averaging the KL divergences, used only when handling lists of distributions.

    Returns
    -------
    float
        The average KL divergence.
    """

    if mu0.ndim == 1:
        mu0 = mu0[np.newaxis, :]
        Sigma0 = Sigma0[np.newaxis, :, :]
    if mu1.ndim == 1:
        mu1 = mu1[np.newaxis, :]
        Sigma1 = Sigma1[np.newaxis, :, :]
    if weights is None:
        weights = np.ones(mu0.shape[0])

    if mu0.shape[0] != mu1.shape[0]:
        if mu1.shape[0] == 1:
            mu1 = np.repeat(mu1, mu0.shape[0], axis=0)
            Sigma1 = np.repeat(Sigma1, mu0.shape[0], axis=0)
        elif mu0.shape[0] == 1:
            mu0 = np.repeat(mu0, mu1.shape[0], axis=0)
            Sigma0 = np.repeat(Sigma0, mu1.shape[0], axis=0)

    divergences = []
    for i in range(mu0.shape[0]):
        mu0_i = mu0[i]
        Sigma0_i = Sigma0[i]
        mu1_i = mu1[i]
        Sigma1_i = Sigma1[i]

        k = mu0_i.shape[0]
        Sigma1_inv = la.inv(Sigma1_i)
        trace_term = np.trace(Sigma1_inv @ Sigma0_i)
        mean_diff = mu1_i - mu0_i
        quadratic_term = mean_diff.T @ Sigma1_inv @ mean_diff
        logdet_term = np.log(np.linalg.det(Sigma1_i) / np.linalg.det(Sigma0_i))
        kl_div = 0.5 * (trace_term + quadratic_term - k + logdet_term)
        divergences.append(kl_div)

    weighted_average = np.average(divergences, weights=weights)
    return weighted_average


def symmetric_kl_divergence_gaussian(
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the symmetric Kullback-Leibler divergence between two
    multivariate Gaussian distributions or sets of distributions.
    Can handle:
    - Single distribution against single distribution.
    - List of distributions against a single distribution.
    - List of distributions against a list of distributions of the same length.

    Parameters
    ----------
    mu0 : np.ndarray
        Mean of the first Gaussian distribution(s).
    Sigma0 : np.ndarray
        Covariance matrix of the first Gaussian distribution(s).
    mu1 : np.ndarray
        Mean of the second Gaussian distribution(s).
    Sigma1 : np.ndarray
        Covariance matrix of the second Gaussian distribution(s).
    weights : np.ndarray, optional
        Weights for averaging the KL divergences, used only when handling lists of distributions.

    Returns
    -------
    float
        The average symmetric KL divergence.
    """

    kl_div_0_to_1 = kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1, weights)
    kl_div_1_to_0 = kl_divergence_gaussian(mu1, Sigma1, mu0, Sigma0, weights)

    symmetric_kl = kl_div_0_to_1 + kl_div_1_to_0
    return symmetric_kl


def mahalanobis_divergence(
    A: np.ndarray, B: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute the weighted Mahalanobis divergence between two matrices or sets of matrices A and B.
    Can handle:
    - Single matrices against single matrices.
    - Lists of matrices against lists of matrices.

    Parameters
    ----------
    A : np.ndarray
        The first input matrix or set of matrices.
    B : np.ndarray
        The second input matrix or set of matrices, must be of the same shape as A.
    weights : np.ndarray, optional
        Weights for each matrix pair, default is uniform weights.

    Returns
    -------
    float
        The weighted average Mahalanobis divergence if multiple matrices are provided,
        otherwise the divergence for a single pair.
    """
    A, B = match_matrices(A, B)

    if weights is None:
        weights = np.ones(A.shape[0])

    divergences = [
        np.trace(
            np.dot(A[i], A[i].T)
            - 2 * np.dot(A[i], B[i].T)
            + np.dot(B[i], B[i].T)
        )
        for i in range(A.shape[0])
    ]
    return np.average(divergences, weights=weights)


def stein_loss(
    S: np.ndarray, Sigma: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Stein's loss for the estimator S and the true covariance
    matrix Sigma.

    Stein's loss is defined as:
        L(S, Sigma) = tr(SSigma^-1) - log(det(SSigma^-1)) - n
    where n is the dimension of S and Sigma, and Sigma^-1 is the
    inverse of Sigma.

    Can handle:
    - Single covariance matrix against single covariance matrix.
    - Lists of covariance matrices against lists of covariance matrices.

    Parameters
    ----------
    S : np.ndarray
        The estimated covariance matrix or set of matrices.
    Sigma : np.ndarray
        The true covariance matrix or set of matrices, each must be invertible.
    weights : np.ndarray, optional
        Weights for each matrix pair, default is uniform weights.

    Returns
    -------
    float
        The weighted average Stein's loss if multiple matrices are provided,
        otherwise the loss for a single pair.
    """
    Sigma, S = match_matrices(Sigma, S)

    if weights is None:
        weights = np.ones(S.shape[0])

    losses = []
    for i in range(S.shape[0]):
        if la.det(Sigma[i]) == 0:
            raise ValueError("Sigma must be invertible.")
        Sigma_inv = la.inv(Sigma[i])
        SSigma_inv = np.dot(S[i], Sigma_inv)
        trace_part = np.trace(SSigma_inv)
        sign, log_det_part = la.slogdet(SSigma_inv)
        if sign <= 0:
            continue  # Skip non-positive definite cases
        n = S[i].shape[0]
        losses.append(trace_part - log_det_part - n)

    return np.average(losses, weights=weights) if losses else float("nan")


def von_neumann_relative_entropy(
    Sigma: np.ndarray, S: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute the von Neumann relative entropy between two positive definite
    matrices Sigma and S.

    The von Neumann relative entropy is defined as:
        H(Sigma, S) = tr(Sigma log(Sigma) - Sigma log(S)) - tr(Sigma - S)
    where log denotes the matrix logarithm.

    Can handle:
    - Single matrices against single matrices.
    - Lists of matrices against lists of matrices.

    Parameters
    ----------
    Sigma : np.ndarray
        The first positive definite matrix or set of matrices.
    S : np.ndarray
        The second positive definite matrix or set of matrices, both must be positive definite.
    weights : np.ndarray, optional
        Weights for each matrix pair, default is uniform weights.

    Returns
    -------
    float
        The weighted average von Neumann relative entropy if multiple
        matrices are provided, otherwise the entropy for a single pair.
    """

    Sigma, S = match_matrices(Sigma, S)

    if weights is None:
        weights = np.ones(Sigma.shape[0])

    entropies = []
    for i in range(Sigma.shape[0]):
        if la.det(Sigma[i]) <= 0 or la.det(S[i]) <= 0:
            raise ValueError("Both matrices must be positive definite.")
        log_Sigma = logm(Sigma[i])
        log_S = logm(S[i])
        entropy = np.trace(
            np.dot(Sigma[i], log_Sigma) - np.dot(Sigma[i], log_S)
        ) - np.trace(Sigma[i] - S[i])
        entropies.append(entropy)

    return np.average(entropies, weights=weights)


def logdet_divergence(
    A: np.ndarray, B: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute the weighted log-determinant divergence between two positive
    definite matrices A and B.

    The log-determinant divergence is defined as:
        D_LD(A, B) = tr(A B^-1) - log(det(A B^-1)) - n
    where B^-1 is the inverse of B, n is the dimension of A and B, and
    det denotes determinant.

    Can handle:
    - Single matrices against single matrices.
    - Lists of matrices against lists of matrices.

    Parameters
    ----------
    A : np.ndarray
        The first set of positive definite matrices.
    B : np.ndarray
        The second set of positive definite matrices, must be of the same shape as A.
    weights : np.ndarray, optional
        Weights for each matrix pair, default is uniform weights.

    Returns
    -------
    float
        The weighted average log-determinant divergence if multiple matrices are provided.
    """

    A, B = match_matrices(A, B)

    if weights is None:
        weights = np.ones(A.shape[0])

    divergences = []
    for i in range(A.shape[0]):
        if la.det(A[i]) <= 0 or la.det(B[i]) <= 0:
            raise ValueError("Both matrices must be positive definite.")
        X = la.solve(B[i], A[i])
        trace_X = np.trace(X)
        sign, log_det_X = np.linalg.slogdet(X)
        if sign <= 0:
            continue
        n = A[i].shape[0]
        divergence = trace_X - log_det_X - n
        divergences.append(divergence)

    return np.average(divergences, weights=weights)


def loss_harmonic(
    A: np.ndarray, B: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the weighted harmonic loss between two matrices or sets of matrices A and B.
    Handles:
    - Single matrices against single matrices.
    - A list of matrices against a single matrix.
    - A list of matrices against a list of matrices of the same length.

    Parameters
    ----------
    A : np.ndarray
        The first matrix or set of matrices.
    B : np.ndarray
        The second matrix or set of matrices, must be of the same shape as A if both are sets.
    weights : np.ndarray, optional
        Weights for each matrix pair, default is uniform weights if multiple matrices are provided.

    Returns
    -------
    float
        The average or weighted average harmonic distance.
    """

    A, B = match_matrices(A, B)

    if weights is None:
        weights = np.ones(A.shape[0])

    distances = [
        la.norm(la.inv(A[i]) - la.inv(B[i]), ord="fro")
        for i in range(A.shape[0])
    ]
    return np.average(distances, weights=weights)


def loss_logeuclidean(
    A: np.ndarray, B: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the weighted log-Euclidean loss between two matrices or sets of matrices A and B.
    Handles:
    - Single matrices against single matrices.
    - A list of matrices against a single matrix.
    - A list of matrices against a list of matrices of the same length.

    Parameters
    ----------
    A : np.ndarray
        The first matrix or set of matrices.
    B : np.ndarray
        The second matrix or set of matrices, must be of the same shape as A if both are sets.
    weights : np.ndarray, optional
        Weights for each matrix pair, default is uniform weights if multiple matrices are provided.

    Returns
    -------
    float
        The average or weighted average log-Euclidean distance.
    """
    A, B = match_matrices(A, B)

    if weights is None:
        weights = np.ones(A.shape[0])

    distances = [
        la.norm(logm(A[i]) - logm(B[i]), ord="fro") for i in range(A.shape[0])
    ]
    return np.average(distances, weights=weights)


def match_matrices(A: np.ndarray, B: np.ndarray):
    """
    Match matrices A and B for operations that require matching dimensions.
    This includes promoting 2D arrays to 3D and handling mismatched sets by replication.

    Parameters
    ----------
    A : np.ndarray
        The first matrix or set of matrices.
    B : np.ndarray
        The second matrix or set of matrices.

    Returns
    -------
    tuple
        A tuple of numpy.ndarrays A and B prepared for subsequent operations.

    Raises
    ------
    ValueError
        If the number of matrices in A and B cannot be matched for pairwise computation.
    """
    if A.ndim == 2:
        A = A[np.newaxis, :]
    if B.ndim == 2:
        B = B[np.newaxis, :]

    if A.shape[0] != B.shape[0]:
        if B.shape[0] == 1:
            B = np.repeat(B, A.shape[0], axis=0)
        elif A.shape[0] == 1:
            A = np.repeat(A, B.shape[0], axis=0)
        else:
            raise ValueError(
                f"The number of matrices in A ({A.shape[0]}) and B ({B.shape[0]}) are "
                f"incompatible and cannot be matched for pairwise computation."
            )

    return A, B
