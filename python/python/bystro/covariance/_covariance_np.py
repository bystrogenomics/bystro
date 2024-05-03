"""
This module provides classes for fitting covariance matrices using various
approaches, including empirical, Bayesian, linear shrinkage, and nonlinear
shrinkage methods. These classes extend BaseCovariance, handling missing data
checks and employing advanced statistical methods for optimal estimation.

Classes:
- EmpiricalCovariance: Fits standard sample covariance.
- BayesianCovariance: Implements MAP estimation with priors.
- LinearShrinkageCovariance: Combines empirical covariance with a structured
  estimator.
- NonLinearShrinkageCovariance: Adjusts eigenvalues using a nonlinear
  shrinkage based on sample spectral density.

Dependencies:
- numpy for matrix operations.
- Preprocessed input data as centered, non-missing values.
"""
from typing import Any
import numpy as np
from numpy.typing import NDArray
from bystro.covariance._base_covariance import (
    BaseCovariance,
    _symmeterize_and_warning,
)


class EmpiricalCovariance(BaseCovariance):
    def __init__(self):
        """
        This object just fits the covariance matrix as the standard sample
        covariance matrix. Does not handle missing values
        """
        super().__init__()

    def fit(self, X: NDArray) -> "EmpiricalCovariance":
        """
        This fits a covariance matrix using samples X.

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The centered data

        Returns
        -------
        self : EmpiricalCovariance
            The model

        Raises
        ------
        ValueError:
            A value error will be raised if missing data is found in X (
            np.isnan(X) evaluates to true ), or if X is not an NDArray
        """
        self._test_inputs(X)
        self.N, self.p = X.shape
        XTX = np.dot(X.T, X)
        covariance = XTX / self.N
        self.covariance = _symmeterize_and_warning(covariance)

        return self


class BayesianCovariance(BaseCovariance):
    def __init__(self, prior_options=None):
        """
        This object fits the covariance matrix as the MAP estimator using
        user-defined priors. Does not handle missing values
        """
        super().__init__()
        if prior_options is None:
            prior_options = {}

        self.prior_options = self._fill_prior_options(prior_options)

    def fit(self, X: NDArray[np.float_]) -> "BayesianCovariance":
        """
        This fits a covariance matrix using samples X with MAP estimation.

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        Returns
        -------
        self : BayesianCovariance
            The model

        Raises
        ------
        ValueError:
            A value error will be raised if missing data is found in X (
            np.isnan(X) evaluates to true ), or if X is not an NDArray
        """
        self._test_inputs(X)
        self.N, self.p = X.shape

        p_opts = self.prior_options
        covariance_empirical = np.dot(X.T, X)
        nu = p_opts["iw_params"]["pnu"] + self.p
        cov_prior = p_opts["iw_params"]["sigma"] * np.eye(self.p)
        posterior_cov = cov_prior + covariance_empirical
        posterior_nu = nu + self.N

        covariance = posterior_cov / (posterior_nu + self.p + 1)
        self.covariance = _symmeterize_and_warning(covariance)

        return self

    def _fill_prior_options(
        self, prior_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        This sets the prior options for our inference scheme

        Parameters
        ----------
        prior_options : dict
            The original prior options passed as a dictionary

        Options
        -------
        iw_params : dict,default={'pnu':2,'sigma':1.0}
            pnu : int - nu = p + pnu
            sigma : float>0 - cov = sigma*I_p
        """
        default_options = {
            "iw_params": {"pnu": 2, "sigma": 1.0},
        }
        return {**default_options, **prior_options}


class LinearShrinkageCovariance(BaseCovariance):
    def fit(self, X: NDArray[np.float_]) -> "LinearShrinkageCovariance":
        """
        This fits a covariance matrix using the linear shrinkage approach
        to combine the empirical covariance matrix with a structured
        estimator. This uses the 2004 paper from Ledoit and Wolf

        Parameters
        ----------
        X : np.array-like, (n_samples, n_covariates)
            The centered data

        Returns
        -------
        self : LinearShrinkageCovariance
            The model, with updated covariance attribute

        Raises
        ------
        ValueError:
            A value error will be raised if missing data is found in X (
            np.isnan(X) evaluates to true ), or if X is not an NDArray
        """

        self._test_inputs(X)
        self.N, self.p = X.shape

        S = np.cov(X.T)

        # Compute shrinkage intensity
        evalues = np.linalg.eigvalsh(S)
        lambd = np.mean(evalues)

        temp = [
            np.linalg.norm(np.outer(X[i], X[i]) - S, "fro") ** 2
            for i in range(self.N)
        ]

        b_bar_sq = np.sum(temp) / self.N**2
        d_sq = np.linalg.norm(S - lambd * np.eye(self.p), "fro") ** 2
        b_sq = np.minimum(b_bar_sq, d_sq)
        rho_hat = b_sq / d_sq

        # Shrink covariance matrix
        lambda_bar = np.trace(S) / self.p
        covariance = rho_hat * lambda_bar * np.eye(self.p) + (1 - rho_hat) * S
        self.covariance = _symmeterize_and_warning(covariance)

        return self


class NonLinearShrinkageCovariance(BaseCovariance):
    """
    This method adjusts each eigenvalue of the sample covariance matrix
    individually according to a nonlinear shrinkage formula, based on the
    Hilbert transform. This approach contrasts with linear shrinkage methods
    that apply a uniform adjustment across all eigenvalues. The nonlinear
    technique allows for a more nuanced adjustment that is particularly
    beneficial in scenarios with clusters of eigenvalues, optimizing the
    shrinkage based on local eigenvalue densities. The method assumes a
    sample matrix composed of i.i.d. random variables with mean zero and
    finite 16th moment, and is suitable for large-dimensional data sets where
    the ratio of variables p to n is significant.

    The theoretical basis of this approach is built around the optimal
    estimation of p parameters, which balances between overfitting p^2
    parameters and underfitting with only 1 parameter. The nonlinear shrinkage
    is calculated using an oracle estimator that depends on a sample
    spectral density
    function approximated via kernel estimation using the Epanechnikov kernel.
    This estimation facilitates the derivation of shrinkage intensities for each
    eigenvalue, tailoring the adjustment to improve estimations under various
    data conditions.

    https://www.jstor.org/stable/27028732
    """

    def fit(self, X: NDArray[np.float_]) -> "NonLinearShrinkageCovariance":
        """
        This fits a covariance matrix using the nonlinear shrinkage approach,
        which adjusts the empirical eigenvalues based on asymptotic results.

        Parameters
        ----------
        X : np.array-like, (n_samples, n_covariates)
            The centered data

        Returns
        -------
        self : NonLinearShrinkageCovariance
            The model, with updated covariance attribute

        Raises
        ------
        ValueError:
            A value error will be raised if missing data is found in X (
            np.isnan(X) evaluates to true ), or if X is not an NDArray
        """
        self._test_inputs(X)
        self.N, self.p = X.shape
        N, p = X.shape
        S = np.cov(X.T)
        lambd, u = np.linalg.eigh(S)

        lambd = lambd[np.maximum(0, p - N) :]
        L = np.tile(lambd, (np.minimum(p, N), 1)).T
        h = N ** (-1 / 3)

        H = h * L.T
        x = (L - L.T) / H
        ftilde = 3 / 4 / np.sqrt(5)
        ftilde *= np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)

        Hftemp = (-3 / 10 / np.pi) * x + (3 / 4 / np.sqrt(5) / np.pi) * (
            1 - x**2 / 5
        ) * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))

        Hftemp[np.abs(x) == np.sqrt(5)] = (-3 / 10 / np.pi) * x[
            np.abs(x) == np.sqrt(5)
        ]

        Hftilde = np.mean(Hftemp / H, axis=1)

        if p <= N:
            dtilde = lambd / (
                (np.pi * (p / N) * lambd * ftilde) ** 2
                + (1 - (p / N) - np.pi * (p / N) * lambd * Hftilde) ** 2
            )
        else:
            Hftilde0 = (
                (1 / np.pi)
                * (
                    3 / 10 / h**2
                    + 3
                    / 4
                    / np.sqrt(5)
                    / h
                    * (1 - 1 / 5 / h**2)
                    * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))
                )
                * np.mean(1 / lambd)
            )
            dtilde0 = 1 / (np.pi * (p - N) / N * Hftilde0)
            dtilde1 = lambd / (
                np.pi**2 * lambd**2 * (ftilde**2 + Hftilde**2)
            )
            dtilde = np.concatenate([dtilde0 * np.ones((p - N)), dtilde1])

        covariance = np.dot(np.dot(u, np.diag(dtilde)), u.T)
        self.covariance = _symmeterize_and_warning(covariance)

        return self
