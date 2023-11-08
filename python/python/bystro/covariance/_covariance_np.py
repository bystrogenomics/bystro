"""
This implements two objects, the empirical covariance estimator used as a 
baseline comparison method. It also implements BayesianCovariance, which
implements MAP estimation using several common priors.

Objects
-------
EmpiricalCovariance(BaseCovariance)
    This object just fits the covariance matrix as the standard sample
    covariance matrix

BayesianCovariance(BaseCovariance):
    This object fits the covariance matrix as the MAP estimator using
    user-defined priors.
"""
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
        self.N, self.p = X.shape

        p_opts = self.prior_options
        covariance_empirical = np.dot(X.T, X)
        nu = p_opts["iw_params"]["pnu"] + self.p
        cov_prior = p_opts["iw_params"]["sigma"] * np.eye(self.p)
        posterior_cov = cov_prior + covariance_empirical
        posterior_nu = nu + self.N

        self.covariance = posterior_cov / (posterior_nu + self.p + 1)

        return self

    def _fill_prior_options(self, prior_options: dict[str, Any]) -> dict[str, Any]:
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
