"""
This implements the standard version of probabilistic PCA found in Chapter
12 of Bishop (2006). It has an analytic solution making this method very
quick to fit. Will serve as a baseline comparison for many of our methods.

Objects
-------
PPCAanalytic(n_components=2)
    The plain vanilla probabilistic PCA model with an analytic formula that
    is estimated via maximum likelihood.

Methods
-------
None
"""
import numpy as np
import numpy.linalg as la
from bystro.supervised_ppca._base import BaseGaussianFactorModel


class PPCAanalytic(BaseGaussianFactorModel):
    def __init__(self, n_components=2):
        """
        Analytic PPCA solution as described by Bishop

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality
        """
        super().__init__(n_components=n_components)

    def __repr__(self):
        out_str = "PPCA_analytic object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        Returns
        -------
        self : object
            The model
        """
        N, self.p = X.shape
        L, p = self.n_components, self.p

        U, s, V = la.svd(X, full_matrices=False)
        evals = s ** 2 / (N - 1)

        var = 1.0 / (p - L) * (np.sum(evals) - np.sum(evals[:L]))

        L_m = np.diag((evals[:L] - np.ones(L) * var) ** 0.5)
        W = np.dot(V[:L].T, L_m)
        self._save_variables([W, var])

    def get_covariance(self):
        """
        Gets the covariance matrix

        Sigma = W^TW + sigma2*I

        Parameters
        ----------
        None

        Returns
        -------
        covariance : np.array-like(p,p)
            The covariance matrix
        """
        covariance = np.dot(self.W_.T, self.W_) + self.sigma2_ * np.eye(self.p)
        return covariance

    def _save_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of variables saved

        Sets
        ----
        W_ : np.array-like,(n_components,p)
            The loadings

        sigma2_ : float
            The isotropic variance
        """
        self.W_ = trainable_variables[0].T
        self.sigma2_ = trainable_variables[1]
