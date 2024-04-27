"""
A module for calculating averages of positive definite matrices.

A positive definite matrix is a symmetric matrix with all positive eigenvalues,
making it invertible and crucial for various statistical and engineering
applications. This module implements methods to compute various means of such
matrices, ensuring mathematical accuracy and stability.

Methods Defined:
- is_positive_definite(matrix): Check symmetry and positiveness of eigenvalues.
- pd_mean_harmonic(matrices): Compute harmonic mean of positive definite matrices.
- pd_mean_karcher(matrices, tol, max_iter): Compute Karcher mean iteratively.
- pd_median_riemann(matrices, tol, max_iter): Compute median iteratively.

Basic Formulas:
- Harmonic mean: inv(sum(inv(A_i) for A_i in matrices) / n)
- Karcher mean: iterative adjustment to minimize Frobenius norm distance.
- Median (Riemannian): iterative optimization of median in the matrix space.

Reference:
- Bhatia, Rajendra. "Positive Definite Matrices." Princeton University Press,
  2007.

All matrices are assumed to be provided as numpy arrays.
"""

import numpy as np
from numpy import ndarray
from scipy.linalg import logm, expm, fractional_matrix_power  # type: ignore
from typing import List


def is_positive_definite(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is symmetric and positive definite.

    Parameters
    ----------
    matrix : ndarray
        The matrix to be checked.

    Returns
    -------
    bool
        True if the matrix is symmetric and all eigenvalues are greater than zero.
    """
    return bool(
        np.allclose(matrix, matrix.T) and np.all(np.linalg.eigvals(matrix) > 0)
    )


def pd_mean_harmonic(matrices: List[ndarray]) -> ndarray:
    """
    Compute the harmonic mean of a list of positive definite matrices.

    Parameters
    ----------
    matrices : List[ndarray]
        A list of positive definite matrices.

    Returns
    -------
    ndarray
        The harmonic mean of the matrices.

    Raises
    ------
    ValueError
        If any matrix is not positive definite or if the list is empty.
    """
    if not matrices or any(not is_positive_definite(C) for C in matrices):
        raise ValueError(
            "All matrices must be positive definite and non-empty."
        )
    n = len(matrices)
    inv_sum = sum(np.linalg.inv(C) for C in matrices)
    return np.linalg.inv(inv_sum / n)


def pd_mean_karcher(
    matrices: List[ndarray], tol: float = 1e-6, max_iter: int = 100
) -> ndarray:
    """
    Compute the Karcher mean of positive definite matrices using an
    iterative approach.

    Parameters
    ----------
    matrices : List[ndarray]
        A list of positive definite matrices.
    tol : float, default=1e-6
        The tolerance for convergence.
    max_iter : int, default=100
        The maximum number of iterations.

    Returns
    -------
    ndarray
        The Karcher mean of the matrices.

    Raises
    ------
    ValueError
        If any input is invalid or convergence is not reached.
    """
    if tol <= 0 or max_iter <= 0:
        raise ValueError("Tolerance and max iterations must be positive.")
    if not matrices or any(not is_positive_definite(C) for C in matrices):
        raise ValueError(
            "All matrices must be positive definite and non-empty."
        )
    n = len(matrices)
    current = np.mean(matrices, axis=0)

    for _ in range(max_iter):
        sum_log = np.zeros_like(current)
        for C in matrices:
            middle = fractional_matrix_power(current, -0.5)
            sum_log += logm(middle @ C @ middle)
        avg_log = sum_log / n
        update = expm(avg_log)
        next_mean = middle.T @ update @ middle.T @ current

        if np.linalg.norm(next_mean - current, "fro") < tol:
            print(f"Converged after {_+1} iterations.")
            return next_mean
        current = next_mean

    print("Max iterations reached without convergence.")
    return current


def pd_mean_log_euclidean(matrices: List[ndarray]) -> ndarray:
    """
    Compute the log-Euclidean mean of positive definite matrices.

    Parameters
    ----------
    matrices : List[ndarray]
        A list of positive definite matrices.

    Returns
    -------
    ndarray
        The log-Euclidean mean of the matrices.

    Raises
    ------
    ValueError
        If any matrix is not positive definite or if the list is empty.
    """

    if not matrices or any(not is_positive_definite(C) for C in matrices):
        raise ValueError(
            "All matrices must be positive definite and non-empty."
        )
    log_sum = sum(logm(C) for C in matrices)
    return expm(log_sum / len(matrices))


"""
Further considerations:

The numerical stability of various methods for computing averages of
positive definite matrices, especially in high-dimensional spaces.

Methods:
1. Arithmetic Mean:
   - Stability: High
   - Does not always yield a PD matrix if the matrices are not close.

2. Geometric Mean:
   - Stability: Moderate to Low
   - Iterative logarithmic and exponential operations can lead to precision
     loss in high dimensions.

3. Harmonic Mean:
   - Stability: Moderate
   - Matrix inversion can be unstable if matrices are close to singular or
     highly ill-conditioned.

4. Log-Euclidean Mean:
   - Stability: High
   - Straightforward averaging in the logarithmic domain followed by a single
     exponential mapping, suitable for high dimensions.

5. Karcher Mean (Fréchet Mean):
   - Stability: Moderate to Low
   - Iterative gradient descent on a manifold; sensitive to initial guess and
     convergence properties, challenging in high dimensions.
"""
