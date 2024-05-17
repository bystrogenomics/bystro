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
    A: np.ndarray, B: np.ndarray, p: float
) -> float:
    """
    Compute the Bregman divergence based on the Schatten p-norm between
    two matrices A and B.

    The Bregman divergence for Schatten norms is defined as:
        D_p(A, B) = ||A||_p^p - ||B||_p^p - p * tr((B^p-1)^(T) * (A - B))
    where ||.||_p denotes the Schatten p-norm.

    Parameters
    ----------
    A : np.ndarray
        The first input matrix.
    B : np.ndarray
        The second input matrix, must be of the same shape as A.
    p : float
        The order of the Schatten norm used in the divergence calculation.

    Returns
    -------
    float
        The Bregman divergence based on the Schatten p-norm between A and B.
    """
    norm_B_p = schatten_norm(B, p)
    norm_A_p = schatten_norm(A, p)
    U, Sigma, Vt = la.svd(B, full_matrices=False)
    B_norm_p_minus_1 = np.dot(U, np.diag(np.power(Sigma, p - 1))) @ Vt
    trace_term = np.trace(B_norm_p_minus_1.T @ (A - B))
    return norm_A_p**p - norm_B_p**p - p * trace_term


def frobenius_loss(A: np.ndarray, B: np.ndarray):
    """
    Calculate the Frobenius loss between two matrices A and B.

    The Frobenius loss is defined as the Frobenius norm of the difference
    between A and B:
        L_F(A, B) = ||A - B||_F
    where ||.||_F denotes the Frobenius norm.

    Parameters
    ----------
    A : np.ndarray
        The first input matrix.
    B : np.ndarray
        The second input matrix, must be of the same shape as A.

    Returns
    -------
    floating[Any]
        The Frobenius loss between A and B.
    """
    difference = A - B
    return la.norm(difference, "fro")


def kl_divergence_gaussian(
    mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray, Sigma1: np.ndarray
) -> float:
    """
    Compute the Kullback-Leibler divergence between two multivariate
    Gaussian distributions.

    Parameters
    ----------
    mu0 (np.ndarray):
        Mean of the first Gaussian distribution.

    Sigma0 (np.ndarray):
        Covariance matrix of the first Gaussian distribution.

    mu1 (np.ndarray):
        Mean of the second Gaussian distribution.

    Sigma1 (np.ndarray):
        Covariance matrix of the second Gaussian distribution.

    Returns
    -------
    kl_div: float
        The KL divergence from Gaussian N(mu0, Sigma0) to N(mu1, Sigma1).
    """
    k = mu0.shape[0]
    Sigma1_inv = la.inv(Sigma1)
    trace_term = np.trace(Sigma1_inv @ Sigma0)
    mean_diff = mu1 - mu0
    quadratic_term = mean_diff.T @ Sigma1_inv @ mean_diff
    logdet_term = np.log(np.linalg.det(Sigma1) / np.linalg.det(Sigma0))
    kl_div = 0.5 * (trace_term + quadratic_term - k + logdet_term)
    return kl_div


def symmetric_kl_divergence_gaussian(
    mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray, Sigma1: np.ndarray
) -> float:
    """
    Compute the symmetric Kullback-Leibler divergence between two multivariate Gaussian distributions.

    Parameters
    ----------
    mu0 (np.ndarray):
        Mean of the first Gaussian distribution.

    Sigma0 (np.ndarray):
        Covariance matrix of the first Gaussian distribution.

    mu1 (np.ndarray):
        Mean of the second Gaussian distribution.

    Sigma1 (np.ndarray):
        Covariance matrix of the second Gaussian distribution.

    Returns
    -------
    symmartic_kl float:
        The symmetric KL divergence between Gaussian N(mu0, Sigma0) and
        N(mu1, Sigma1).
    """
    kl_div_0_to_1 = kl_divergence_gaussian(mu0, Sigma0, mu1, Sigma1)
    kl_div_1_to_0 = kl_divergence_gaussian(mu1, Sigma1, mu0, Sigma0)
    symmetric_kl = kl_div_0_to_1 + kl_div_1_to_0
    return symmetric_kl


def mahalanobis_divergence(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the Mahalanobis divergence between two matrices A and B.

    The Mahalanobis divergence is defined as:
        M(A, B) = tr(AA' - 2AB' + BB')
    where A' and B' denote the transpose of A and B, respectively.

    Parameters
    ----------
    A : np.ndarray
        The first input matrix.
    B : np.ndarray
        The second input matrix, must be of the same shape as A.

    Returns
    -------
    float
        The Mahalanobis divergence between A and B.
    """
    return np.trace(np.dot(A, A) - 2 * np.dot(A, B) + np.dot(B, B))


def stein_loss(S: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Calculate Stein's loss for the estimator S and the true covariance
    matrix Sigma.

    Stein's loss is defined as:
        L(S, Sigma) = tr(SSigma^-1) - log(det(SSigma^-1)) - n
    where n is the dimension of S and Sigma, and Sigma^-1 is the
    inverse of Sigma.

    Parameters
    ----------
    S : np.ndarray
        The estimated covariance matrix.
    Sigma : np.ndarray
        The true covariance matrix, must be invertible.

    Returns
    -------
    float
        Stein's loss for the given matrices.
    """
    if la.det(Sigma) == 0:
        raise ValueError("Sigma must be invertible.")
    Sigma_inv = la.inv(Sigma)
    SSigma_inv = np.dot(S, Sigma_inv)
    trace_part = np.trace(SSigma_inv)
    _, log_det_part = la.slogdet(SSigma_inv)
    n = S.shape[0]
    return trace_part - log_det_part - n


def von_neumann_relative_entropy(Sigma: np.ndarray, S: np.ndarray) -> float:
    """
    Compute the von Neumann relative entropy between two positive definite
    matrices Sigma and S.

    The von Neumann relative entropy is defined as:
        H(Sigma, S) = tr(Sigma log(Sigma) - Sigma log(S)) - tr(Sigma - S)
    where log denotes the matrix logarithm.

    Parameters
    ----------
    Sigma : np.ndarray
        The first positive definite matrix.
    S : np.ndarray
        The second positive definite matrix, both matrices must be
        positive definite.

    Returns
    -------
    float
        The von Neumann relative entropy between Sigma and S.
    """
    if la.det(Sigma) <= 0 or la.det(S) <= 0:
        raise ValueError("Both matrices must be positive definite.")
    log_Sigma = logm(Sigma)
    log_S = logm(S)
    Sigma_log_Sigma = np.dot(Sigma, log_Sigma)
    Sigma_log_S = np.dot(Sigma, log_S)
    trace_Sigma_log_Sigma = np.trace(Sigma_log_Sigma)
    trace_Sigma_log_S = np.trace(Sigma_log_S)
    return trace_Sigma_log_Sigma - trace_Sigma_log_S - np.trace(Sigma - S)


def logdet_divergence(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the log-determinant divergence between two positive
    definite matrices A and B.

    The log-determinant divergence is defined as:
        D_LD(A, B) = tr(A B^-1) - log(det(A B^-1)) - n
    where B^-1 is the inverse of B, n is the dimension of A and B, and
    det denotes determinant.

    Parameters
    ----------
    A : np.ndarray
        The first positive definite matrix.
    B : np.ndarray
        The second positive definite matrix, both matrices must be
        positive definite.

    Returns
    -------
    float
        The log-determinant divergence between A and B.
    """
    if np.linalg.det(A) <= 0 or np.linalg.det(B) <= 0:
        raise ValueError("Both matrices must be positive definite.")
    X = np.linalg.solve(B, A)
    trace_X = np.trace(X)
    sign, log_det_X = np.linalg.slogdet(X)
    if sign <= 0:
        raise ValueError("Resulting matrix X must have a positive determinant.")
    n = A.shape[0]
    return trace_X - log_det_X - n


def loss_harmonic(A: np.ndarray, B: np.ndarray):
    distance = la.norm(la.inv(A) - la.inv(B), ord="fro")
    return distance


def loss_logeuclidean(A: np.ndarray, B: np.ndarray):
    distance = la.norm(logm(A) - logm(B), ord="fro")
    return distance
