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
from numpy.typing import NDArray


class PPCAanalytic(BaseGaussianFactorModel):
    """
    Analytic PPCA solution as described by Bishop

    Parameters
    ----------
    n_components : int,default=2
        The latent dimensionality
    """

    def __init__(self, n_components: int = 2):
        super().__init__(n_components=n_components)
        self.W_: NDArray[np.float_] | None = None
        self.p: int | None = None
        self.sigma2_: np.float_ | None = None

    def __repr__(self) -> str:
        return f"PPCAAnalytic(n_components={self.n_components})"

    def fit(self, X: NDArray[np.float_]) -> "PPCAanalytic":
        """
        Fits a model given covariates X

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        Returns
        -------
        self : PPCAanalytic
            The model
        """
        N, self.p = X.shape
        L, p = self.n_components, self.p

        U, s, V = la.svd(X, full_matrices=False)
        eigenvals = s**2 / (N - 1)

        var = 1.0 / (p - L) * (np.sum(eigenvals) - np.sum(eigenvals[:L]))

        L_m = np.diag((eigenvals[:L] - np.ones(L) * var) ** 0.5)
        W: NDArray[np.float_] = np.dot(V[:L].T, L_m)
        self._store_instance_variables((W, var))

        return self

    def get_covariance(self) -> NDArray[np.float_]:
        """
        Gets the covariance matrix

        Sigma = W^TW + sigma2*I

        Parameters
        ----------
        None

        Returns
        -------
        covariance : NDArray,(p,p)
            The covariance matrix
        """
        if self.W_ is None or self.sigma2_ is None or self.p is None:
            raise ValueError("Model has not been fit yet")

        return np.dot(self.W_.T, self.W_) + self.sigma2_ * np.eye(self.p)

    def get_noise(self):
        """
        Returns the observational noise as a diagonal matrix

        Parameters
        ----------
        None

        Returns
        -------
        Lambda : NDArray,(p,p)
            The observational noise
        """
        if self.sigma2_ is None or self.p is None:
            raise ValueError("Model has not been fit yet")

        return self.sigma2_ * np.eye(self.p)

    def _store_instance_variables(
        self, trainable_variables: tuple[NDArray[np.float_], np.float_]
    ) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of variables saved

        Sets
        ----
        W_ : NDArray,(n_components,p)
            The loadings

        sigma2_ : float
            The isotropic variance
        """
        self.W_ = trainable_variables[0].T
        self.sigma2_ = trainable_variables[1]

    def _initialize_save_losses(self):
        pass

    def _save_losses(self):
        pass

    def _test_inputs(self):
        pass

    def _transform_training_data(self):
        pass

    def _save_variables(self):
        pass
