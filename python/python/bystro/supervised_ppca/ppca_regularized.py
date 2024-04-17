import numpy as np
from numpy.typing import NDArray
import numpy.linalg as la

from bystro.supervised_ppca._base import BaseGaussianFactorModel
from bystro.covariance._covariance_np import (
    BayesianCovariance,
    LinearShrinkageCovariance,
    NonLinearShrinkageCovariance,
)


class PPCARegularized(BaseGaussianFactorModel):
    """
    Analytic PPCA solution as described by Bishop

    Parameters
    ----------
    n_components : int,default=2
        The latent dimensionality
    """

    def __init__(self, n_components: int = 2, regularization_options=None):
        super().__init__(n_components=n_components)
        if regularization_options is None:
            regularization_options = {}
        self.regularization_options = self._fill_regularization_options(
            regularization_options
        )
        self.W_: NDArray[np.float_] | None = None
        self.p: int | None = None
        self.sigma2_: np.float_ | None = None

    def fit(self, X: NDArray[np.float_]) -> "PPCARegularized":
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
        L = self.n_components
        regularization_options = self.regularization_options["method"]

        if regularization_options["method"] == "LinearShrinkage":
            model_cov = LinearShrinkageCovariance()
        elif regularization_options["method"] == "NonLinearShrinkage":
            model_cov = NonLinearShrinkageCovariance()
        elif regularization_options["method"] == "Bayesian":
            model_cov = BayesianCovariance(
                regularization_options["prior_options"]
            )
        else:
            raise ValueError(
                "Unrecognized regularization option %s"
                % regularization_options["method"]
            )
        model_cov.fit(X)
        cov = model_cov.covariance

        eigenvalues, eigenvectors = la.eigh(cov)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]

        sigma2 = np.mean(eigenvalues[L:])
        W = (
            eigenvectors[:, :L]
            * np.sqrt(eigenvalues[:L] - sigma2)[:, np.newaxis]
        )
        W = W.T

        self._store_instance_variables((W, sigma2))

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

    def _fill_regularization_options(self, regularization_options):
        default_options = {
            "method": "LinearShrinkage",
            "prior_options": {"iw_params": {"pnu": 2, "sigma": 3}},
        }
        rops = {**default_options, **regularization_options}
        return rops
