"""
This implements two objects, the empirical covariance estimator used as a 
baseline comparison method. It also implements BayesianCovariance, which
implements MAP estimation using several common priors.

Objects
-------
EmpiricalCovariance(BaseCovariance)
    

"""
import numpy as np
from ._base_covariance import BaseCovariance
from ..utils._misc import fill_dict


class EmpiricalCovariance(BaseCovariance):
    def __init__(self):
        """
        This object just fits the covariance matrix as the standard sample
        covariance matrix
        """
        super().__init__()

    def fit(self, X):
        """
        This fits a covariance matrix using samples X.

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data
        """
        self.N, self.p = X.shape
        XTX = np.dot(X.T, X)
        self.covariance = XTX / self.N


class BayesianCovariance(BaseCovariance):
    # https://arxiv.org/pdf/1408.4050.pdf#:~:text=Bayesian%20estimation%20of%20a%20covariance,prior%20implemented%20in%20Bayesian%20software.

    def __init__(self, prior_options={}):
        """
        This object fits the covariance matrix as the MAP estimator using
        user-defined priors.
        """
        super().__init__()

        self.prior_options = self._fill_prior_options(prior_options)

    def fit(self, X):
        """
        This fits a covariance matrix using samples X with MAP estimation.

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data
        """
        self.N, self.p = X.shape
        p_opts = self.prior_options
        covariance_empirical = np.dot(X.T, X)
        if p_opts["type"] == "inverse-wishart":
            nu = p_opts["iw_params"]["pnu"] + self.p
            cov_prior = p_opts["iw_params"]["sigma"] * np.eye(self.p)
            posterior_cov = cov_prior + covariance_empirical
            posterior_nu = nu + self.N
            self.covariance = posterior_cov / (posterior_nu + self.p + 1)
        else:
            raise ValueError("Type %s not implemented" % p_opts["type"])

    def _fill_prior_options(self, prior_options):
        """
        This sets the prior options for our inference scheme

        Parameters
        ----------
        prior_options : dict
            The original prior options passed as a dictionary

        Options
        -------
        type : str, default='inverse-wishart'
            The type of prior. Currently one lol

        iw_params : dict,default={'pnu':2,'sigma':1.0}
            pnu : int - nu = p + pnu
            sigma : float>0 - cov = sigma*I_p
        """
        default_options = {
            "type": "inverse-wishart",
            "iw_params": {"pnu": 2, "sigma": 1.0},
        }
        return fill_dict(prior_options, default_options)
