from typing import Any
import numpy as np
from numpy.typing import NDArray
from bystro.covariance._base_covariance import BaseCovariance


class EmpiricalCovariance(BaseCovariance):
    def __init__(self):
        """
        This object just fits the covariance matrix as the standard sample
        covariance matrix
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
        """
        self._test_inputs(X)
        self.N, self.p = X.shape
        XTX = np.dot(X.T, X)
        self.covariance = XTX / self.N

        return self


class BayesianCovariance(BaseCovariance):
    def __init__(self, prior_options=None):
        """
        This object fits the covariance matrix as the MAP estimator using
        user-defined priors.
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
        """
        self._test_inputs(X)
        self.N, self.p = X.shape

        p_opts = self.prior_options
        covariance_empirical = np.dot(X.T, X)
        nu = p_opts["iw_params"]["pnu"] + self.p
        cov_prior = p_opts["iw_params"]["sigma"] * np.eye(self.p)
        posterior_cov = cov_prior + covariance_empirical
        posterior_nu = nu + self.N

        self.covariance = posterior_cov / (posterior_nu + self.p + 1)

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
        self._test_inputs(X)
        self.N, self.p = X.shape

        S = np.cov(X.T)

        # Compute shrinkage intensity
        evalues = np.linalg.eigvalsh(S)
        lambd = np.mean(evalues)

        temp = [
            np.linalg.norm(np.outer(X[:, i], X[:, i]) - S, "fro") ** 2
            for i in range(self.N)
        ]

        b_bar_sq = np.sum(temp) / self.N**2
        d_sq = np.linalg.norm(S - lambd * np.eye(self.p), "fro") ** 2
        b_sq = np.minimum(b_bar_sq, d_sq)
        rho_hat = b_sq / d_sq

        # Shrink covariance matrix
        lambda_bar = np.trace(S) / self.p
        self.covariance = (
            rho_hat * lambda_bar * np.eye(self.p) + (1 - rho_hat) * S
        )

        return self


class NonLinearShrinkageCovariance(BaseCovariance):
    def fit(self, X: NDArray[np.float_]) -> "NonLinearShrinkageCovariance":
        self._test_inputs(X)
        self.N, self.p = X.shape
        N, p = X.shape
        S = np.cov(X.T)
        lambd, u = np.linalg.eigh(S)

        # compute analytical nonlinear shrinkage kernel formula.
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

        self.covariance = np.dot(np.dot(u, np.diag(dtilde)), u.T)

        return self
