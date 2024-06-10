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

LICENSE FROM pyRiemann

https://github.com/pyRiemann/pyRiemann/blob/master/LICENSE

Copyright (c) 2015-2024, authors of pyRiemann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from numpy import ndarray
from scipy.linalg import sqrtm, logm, expm, fractional_matrix_power, eigvalsh  # type: ignore
from typing import List, Optional
import warnings


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


def median_euclid(
    X: ndarray,
    tol: float = 1e-6,
    maxiter: int = 100,
    init: Optional[ndarray] = None,
    weights: Optional[ndarray] = None,
) -> ndarray:
    """
    Compute the Euclidean median (also known as the geometric median) of a set of matrices.

    The Euclidean median of a set of points in Euclidean space is the point minimizing
    the sum of distances to the sample points. This function iteratively adjusts the median
    estimate using a weighted average where the weights are inversely proportional to the
    distances from the current estimate to each point.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features, n_features)
        Array of n_samples observations, each an n_features x n_features matrix.
    tol : float, default=1e-6
        The tolerance for convergence. The algorithm stops when the Frobenius norm of the
        change in the median estimate is less than `tol`.
    maxiter : int, default=100
        Maximum number of iterations for the algorithm.
    init : ndarray of shape (n_features, n_features), optional
        Initial guess for the median. If None, the weighted average of X is used.
    weights : ndarray of shape (n_samples,), optional
        Weights associated with the matrices in X. If None, all points are given equal weight.

    Returns
    -------
    M : ndarray of shape (n_features, n_features)
        The estimated Euclidean median of the input matrices.

    Notes
    -----
    This implementation uses the Euclidean norm (Frobenius norm for matrices) to calculate
    distances between matrices and adjusts the median using a weighted average approach,
    where the weights are inversely proportional to these distances to reduce the influence
    of outliers.
    """

    n_matrices = X.shape[0]
    weights = np.ones(n_matrices) if weights is None else np.array(weights)

    if init is not None:
        M = np.array(init)
    else:
        M = (
            np.sum(weights[:, np.newaxis, np.newaxis] * X, axis=0)
            / weights.sum()
        )

    for _ in range(maxiter):
        distances = np.linalg.norm(X - M, ord="fro", axis=(1, 2))

        adjusted_distances = np.maximum(distances, tol)
        weight_factors = weights / adjusted_distances

        M_new = np.einsum("i,ijk->jk", weight_factors, X) / weight_factors.sum()

        if np.linalg.norm(M - M_new, "fro") < tol:
            break

        M = M_new

    return M_new


def median_riemann(
    X: ndarray,
    tol: float = 1e-6,
    maxiter: int = 100,
    init: Optional[ndarray] = None,
    weights: Optional[ndarray] = None,
    step_size: float = 1,
) -> ndarray:
    """
    Compute the Riemannian median of a set of positive definite matrices.

    The Riemannian median minimizes the sum of distances on the manifold of
    positive definite matrices. The function implements an iterative algorithm
    using the exponential and logarithmic maps at the matrix space.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features, n_features)
        Array of n_samples positive definite matrices, each of dimension n_features x n_features.
    tol : float, default=1e-6
        The tolerance for convergence. The algorithm stops when the change in the median
        estimate is less than `tol`, measured in the Frobenius norm.
    maxiter : int, default=100
        Maximum number of iterations for the algorithm.
    init : ndarray of shape (n_features, n_features), optional
        Initial guess for the median. If None, the weighted average of X is used.
    weights : ndarray of shape (n_samples,), optional
        Weights associated with the matrices in X. If None, all points are given equal weight.
    step_size : float, default=1
        The step size used in the update step of the Riemannian median calculation.

    Returns
    -------
    M : ndarray of shape (n_features, n_features)
        The estimated Riemannian median of the input matrices.

    Raises

    ------
    ValueError
        If any matrix in X is not positive definite.
        If step size is not within (0,2]
    """
    n_matrices, n, _ = X.shape
    if any(not is_positive_definite(C) for C in X):
        raise ValueError("All matrices must be positive definite.")
    if not 0 < step_size <= 2:
        raise ValueError(
            f"Value step_size must be included in (0, 2] (Got {step_size})"
        )
    n_matrices, _, _ = X.shape
    weights = np.ones(n_matrices) if weights is None else np.array(weights)

    if init is not None:
        M = np.array(init)
    else:
        M = (
            np.sum(weights[:, np.newaxis, np.newaxis] * X, axis=0)
            / weights.sum()
        )

    for _ in range(maxiter):
        dists = np.sqrt(
            (np.log(np.array([eigvalsh(a, M) for a in X])) ** 2).sum(axis=-1)
        )
        is_zero = dists == 0
        w = weights[~is_zero] / dists[~is_zero]

        M12 = sqrtm(M)
        Mm12 = np.linalg.inv(M12)
        tangvecs = np.array([logm(Mm12 @ m @ Mm12) for m in X[~is_zero]])

        J = np.einsum("a,abc->bc", w / np.sum(w), tangvecs)
        M = M12 @ expm(step_size * J) @ M12

        crit = np.linalg.norm(J, ord="fro")
        if crit <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return M


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
