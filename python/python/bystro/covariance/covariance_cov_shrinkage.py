"""
Covariance Matrix Estimation with Shrinkage Techniques

This module implements three different shrinkage techniques for covariance 
matrix estimation: Linear Inverse Shrinkage (LIS), Quadratic Inverse 
Shrinkage (QIS), and Geometric Inverse Shrinkage (GIS). These methods are 
designed to improve the estimation of covariance matrices in scenarios 
where the number of features may be comparable to or exceed the number 
of observations.

Linear Inverse Shrinkage (LIS):
--------------------------------
LIS aims to improve the conditioning of the sample covariance matrix by 
shrinking its eigenvalues. The shrinkage intensity is determined by the 
ratio of the number of features to the number of observations. 
Mathematically, the shrunk covariance matrix is computed as:

    S_hat = U * diag(d_i) * U^T

where U is the matrix of eigenvectors, d_i are the shrunk eigenvalues 
calculated from the original eigenvalues λ_i of the sample covariance 
matrix, and the shrinkage targets are based on the overall variance.

Quadratic Inverse Shrinkage (QIS):
----------------------------------
QIS extends the LIS by considering a quadratic form of the eigenvalue 
adjustment. It is particularly useful when dealing with outliers or 
heavy-tailed distributions. The shrunk eigenvalues in QIS are calculated 
as:

    d_i = 1 / (c^2 * l_i + (1 - c^2) / p * S(1/l_j))

where c is the concentration ratio (p/n), l_i are the eigenvalues, and p 
is the number of features.

Geometric Inverse Shrinkage (GIS):
----------------------------------
GIS uses a geometric mean of the eigenvalues to determine the shrinkage 
intensity. This method is less sensitive to the specific distribution of 
eigenvalues and provides a balance between the largest and smallest 
variances. The formula for GIS is given by:

    d_i = (l_i * h) / (l_i + h - l_i * h)

where h is a parameter typically depending on the geometric mean of the 
eigenvalues.

These techniques are implemented as classes that inherit from a base 
covariance class, allowing them to be easily integrated with other 
statistical analysis pipelines.

Import Dependencies:
    numpy as np
    numpy.typing as NDArray
    math
    from bystro.covariance._base_covariance import BaseCovariance


This code is a minimally modified version of the code in the CovShrinkage
package (the code differs solely to remove pandas). However, the advantage
of incorporating the package into our framework is that it automatically
inherits all the nice bells and whistles you can do with a covariance matrix
in terms of prediction/imputation with multivariate Gaussians.

LICENSE FROM CovShrinkage

https://github.com/pald22/covShrinkage/blob/main/LICENSE

MIT License

Copyright (c) 2022 Anonymous Panda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from bystro.covariance._base_covariance import (
    BaseCovariance,
    _symmeterize_and_warning,
)
import math


class GeometricInverseShrinkage(BaseCovariance):
    def __init__(self) -> None:
        """
        Initialize the Geometric Inverse Shrinkage covariance estimator.
        """
        super().__init__()

    def fit(self, X: NDArray[np.float64]) -> "GeometricInverseShrinkage":
        """
        Fit the Geometric Inverse Shrinkage model to the given data.

        Parameters
        ----------
        X : NDArray[np.float64]
            Input data with shape (n_samples, n_features).

        Returns
        -------
        self : GeometricInverseShrinkage
            The instance itself.
        """
        covariance = gis(X)
        self.covariance = _symmeterize_and_warning(covariance)
        return self


class LinearInverseShrinkage(BaseCovariance):
    def __init__(self) -> None:
        """
        Initialize the Linear Inverse Shrinkage covariance estimator.
        """
        super().__init__()

    def fit(self, X: NDArray[np.float64]) -> "LinearInverseShrinkage":
        """
        Fit the Linear Inverse Shrinkage model to the given data.

        Parameters
        ----------
        X : NDArray[np.float64]
            Input data with shape (n_samples, n_features).

        Returns
        -------
        self : LinearInverseShrinkage
            The instance itself.
        """
        covariance = lis(X)
        self.covariance = _symmeterize_and_warning(covariance)
        return self


class QuadraticInverseShrinkage(BaseCovariance):
    def __init__(self) -> None:
        """
        Initialize the Quadratic Inverse Shrinkage covariance estimator.
        """
        super().__init__()

    def fit(self, X: NDArray[np.float64]) -> "QuadraticInverseShrinkage":
        """
        Fit the Quadratic Inverse Shrinkage model to the given data.

        Parameters
        ----------
        X : NDArray[np.float64]
            Input data with shape (n_samples, n_features).

        Returns
        -------
        self : QuadraticInverseShrinkage
            The instance itself.
        """
        covariance = qis(X)
        self.covariance = _symmeterize_and_warning(covariance)
        return self


def gis(Y: NDArray[np.float64], k: Optional[int] = None) -> NDArray[np.float64]:
    """
    Compute the Geometric Inverse Shrinkage covariance matrix.

    Parameters
    ----------
    Y : NDArray[np.float64]
        Input data matrix with shape (n_samples, n_features).
    k : int, optional
        Adjustment to the degrees of freedom. Default is None.

    Returns
    -------
    NDArray[np.float64]
        The shrunk covariance matrix.
    """
    N, p = Y.shape
    if N <= p:
        raise ValueError(
            "p must be <= n for the Symmetrized Kullback-Leibler divergence"
        )

    if k is None or math.isnan(k):
        Y = Y - Y.mean(axis=0)
        k = 1

    n = N - k
    c = p / n

    sample = np.dot(Y.T, Y) / n
    sample = (sample + sample.T) / 2

    lambda1, u = np.linalg.eigh(sample)
    lambda1 = lambda1.clip(min=0)
    indices = np.argsort(lambda1)
    lambda1 = lambda1[indices]
    u = u[:, indices]

    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1 : p]

    Lj = np.tile(invlambda, (len(invlambda), 1))
    Lj = Lj.T
    Lj_i = Lj - Lj.T

    num = Lj * Lj_i
    den = Lj_i**2 + Lj**2 * h**2
    theta = np.mean(num / den, axis=0)
    Htheta = np.mean(Lj * Lj * h / den, axis=0)
    Atheta2 = theta**2 + Htheta**2

    deltahat_1 = (1 - c) * invlambda + 2 * c * invlambda * theta
    delta = 1 / (
        (1 - c) ** 2 * invlambda
        + 2 * c * (1 - c) * invlambda * theta
        + c**2 * invlambda * Atheta2
    )

    deltaLIS_1 = np.maximum(deltahat_1, np.min(invlambda))

    temp2 = np.diag((delta / deltaLIS_1) ** 0.5)
    sigmahat = np.dot(np.dot(u, temp2), u.T.conjugate())

    return sigmahat


def lis(Y: NDArray[np.float64], k: Optional[int] = None) -> NDArray[np.float64]:
    """
    Compute the Linear Inverse Shrinkage covariance matrix.

    Parameters
    ----------
    Y : NDArray[np.float64]
        Input data matrix with shape (n_samples, n_features).
    k : int, optional
        Adjustment to the degrees of freedom. Default is None.

    Returns
    -------
    NDArray[np.float64]
        The shrunk covariance matrix.
    """
    N, p = Y.shape
    if N <= p:
        raise ValueError("p must be <= n for Stein's loss")

    if k is None or math.isnan(k):
        Y = Y - np.mean(Y, axis=0)  # demean
        k = 1

    n = N - k  # adjust effective sample size
    c = p / n  # concentration ratio

    sample = np.dot(Y.T, Y) / n
    sample = (sample + sample.T) / 2  # make symmetrical

    lambda1, u = np.linalg.eigh(sample)  # use symmetric decomposition
    lambda1 = np.clip(lambda1, 0, None)  # reset negative values to 0

    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35

    valid_range = max(1, p - n + 1) - 1
    invlambda = 1 / lambda1[valid_range:p]

    # Matrix operations to calculate theta
    Lj = np.tile(invlambda, (len(invlambda), 1))
    Lj = Lj.T
    Lj_i = Lj - Lj.T
    numerator = Lj * Lj_i
    denominator = Lj_i**2 + (Lj**2) * h**2
    theta = np.mean(numerator / denominator, axis=0)

    deltahat_1 = (1 - c) * invlambda + 2 * c * invlambda * theta

    # Ensure no eigenvalue shrinkage below minimum
    deltaLIS_1 = np.maximum(deltahat_1, np.min(invlambda))

    # Reconstruct covariance matrix
    temp2 = np.diag(1 / deltaLIS_1)
    sigmahat = np.dot(np.dot(u, temp2), u.T.conjugate())

    return sigmahat


def qis(Y: NDArray[np.float64], k: Optional[int] = None) -> NDArray[np.float64]:
    """
    Compute the Quadratic Inverse Shrinkage covariance matrix.

    Parameters
    ----------
    Y : NDArray[np.float64]
        Input data matrix with shape (n_samples, n_features).
    k : int, optional
        Adjustment to the degrees of freedom. Default is None.

    Returns
    -------
    NDArray[np.float64]
        The shrunk covariance matrix.
    """
    N, p = Y.shape

    if k is None or math.isnan(k):
        Y = Y - np.mean(Y, axis=0)
        k = 1

    n = N - k
    c = p / n

    sample = np.matmul(Y.T, Y) / n
    sample = (sample + sample.T) / 2

    lambda1, u = np.linalg.eigh(sample)
    lambda1 = np.clip(lambda1, 0, None)

    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35
    invlambda = 1 / lambda1[-min(p, n) :]

    Lj = np.tile(invlambda, (len(invlambda), 1))
    Lj_i = Lj - Lj.T
    Lj_i = Lj_i.T
    Lj = Lj.T

    num = Lj * Lj_i
    den = Lj_i**2 + Lj**2 * h**2
    theta = np.mean(num / den, axis=0)
    Htheta = np.mean(Lj * Lj * h / den, axis=0)
    Atheta2 = theta**2 + Htheta**2

    if p <= n:
        delta = 1 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda))
        delta = np.repeat(delta0, p - n)
        delta = np.concatenate((delta, 1 / (invlambda * Atheta2)))

    deltaQIS = delta * (np.sum(lambda1) / np.sum(delta))

    temp2 = np.diag(deltaQIS)
    sigmahat = np.dot(np.dot(u, temp2), u.T.conjugate())

    return sigmahat
