from typing import Union
import numpy as np
from numpy.typing import NDArray
import numpy.linalg as la

from bystro.supervised_ppca._base import BaseGaussianFactorModel
from bystro.covariance._covariance_np import (
    LinearShrinkageCovariance,
    NonLinearShrinkageCovariance,
)


class PPCARegularized(BaseGaussianFactorModel):
    """
    This implements PPCA but with a regularized covariance matrix. PPCA
    can be viewed as fitting a low rank model to the marginal covariance
    matrix on the data, with a noise term that is calculated based on the
    unexplained variance. This is traditionally done using the empirical
    covariance matrix. However, this estimate of the covariance matrix
    is known to perform poorly in practice. So as an alternative, we
    compute this low rank approximation on regularized covariance matrices
    using a variety of strategies.

    Parameters
    ----------
    n_components : int,default=2
        The latent dimensionality

    regularization options : dict
        The regularization options

    Regularization options: The most important choice is regularization_
    options['method']  which determines which strategy is used for
    shrinkage. Current are linear and NonLinear

    Linear: Honey I shrunk the covariance matrix by Ledoit and wolfe

    NonLinear: Analytical NonLinear Shrinkage of Large Dimensional
        Covariance Matrices by Ledoit and wolf
        https://www.jstor.org/stable/27028732


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
        regularization_options = self.regularization_options

        model_cov: Union[
            LinearShrinkageCovariance, NonLinearShrinkageCovariance
        ]
        if regularization_options["method"] == "LinearShrinkage":
            model_cov = LinearShrinkageCovariance()
        elif regularization_options["method"] == "NonLinearShrinkage":
            model_cov = NonLinearShrinkageCovariance()
        elif regularization_options["method"] == "Bayesian":
            raise ValueError("Currently not supported")
        else:
            raise ValueError(
                "Unrecognized regularization option %s"
                % regularization_options["method"]
            )
        model_cov.fit(X)
        cov = model_cov.covariance

        if cov is not None:
            eigenvalues, eigenvectors = la.eigh(cov)
        else:
            raise ValueError("Covariance matrix is none")
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        eigenvalues = eigenvalues[sorted_idx]

        sigma2 = np.mean(eigenvalues[L:])
        W = eigenvectors[:, :L] * np.sqrt(eigenvalues[:L] - sigma2)

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
