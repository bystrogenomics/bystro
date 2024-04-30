"""
This module reworks the augmented-pca package to implement Probabilistic 
PCA (PPCA) instead of standard PCA. PPCA combines the benefits of a 
generative model with the analytic solution typical of PCA. The 
PPCAadversarial object developed herein allows for mitigating the 
influence of confounding variables in the latent representation.

The implementation facilitates more effective dimensionality reduction in 
the presence of confounders by penalizing their representation in the 
latent variables. Two key parameters, 'mu' and 'eps', control the strength 
of the adversarial component and the conditioning of inverses, 
respectively, enhancing the robustness and stability of the model.
"""
import numpy as np
import numpy.linalg as la

from bystro.supervised_ppca._base import BaseGaussianFactorModel
from numpy.typing import NDArray

from bystro.covariance._covariance_np import (
    LinearShrinkageCovariance,
    NonLinearShrinkageCovariance,
)


class PPCAadversarial(BaseGaussianFactorModel):
    """
    Probabilistic PCA that mitigates the influence of confounding
    variables in the latent representation.

    This class extends the traditional PCA model by incorporating an
    adversarial mechanism that penalizes the representation of confounding
    variables. This approach not only reduces dimensionality but also
    ensures that the resulting components are more representative of the
    true underlying signals rather than artifacts introduced by confounders.

    Parameters
    ----------
    mu : float
        The adversarial strength which controls the penalty for
        representing the confounding variables in the latent representation.
    eps : float
        A small constant added to improve the conditioning of
        matrix inverses during computation.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
    Notes
    -----
    This implementation is based on modifications to the
    'augmented-pca' package by Carson, Talbot and Carlson
    """

    def __init__(
        self,
        n_components: int = 2,
        mu=1.0,
        eps=1e-8,
        regularization_options=None,
    ):
        super().__init__(n_components=n_components)
        if regularization_options is None:
            regularization_options = {}
        self.regularization_options = self._fill_regularization_options(
            regularization_options
        )
        self.mu: float = mu
        self.eps: float = eps
        self.W_: NDArray[np.float_] | None = None
        self.D_: NDArray[np.float_] | None = None
        self.p: int | None = None
        self.sigma2_: np.float_ | None = None

    def fit(
        self, X: NDArray[np.float_], Y: NDArray[np.float_]
    ) -> "PPCAadversarial":
        """
        Fits a model given covariates X

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        Y : NDArray,(n_samples,n_confounders)
            The concommitant data we want to remove

        Returns
        -------
        self : PPCAadversarial
            The model
        """
        regularization_options = self.regularization_options
        self.mean_x = np.mean(X, axis=0)
        self.mean_y = np.mean(Y, axis=0)
        X_dm = X - self.mean_x
        Y_dm = Y - self.mean_y
        N, self.p = X.shape
        p = self.p
        q = Y.shape[1]

        model_cov: LinearShrinkageCovariance | NonLinearShrinkageCovariance
        if regularization_options["method"] == "Empirical":
            B_11 = X_dm.T @ X_dm
            B_12 = X_dm.T @ Y_dm
            diag_reg = self.eps * np.eye(X_dm.shape[1])
            XtX = X_dm.T @ X_dm + diag_reg
            B_22 = B_12.T @ la.solve(XtX, B_12)
        elif regularization_options["method"] == "NonLinearShrinkage":
            if q == 1:
                XX = np.zeros((N, p + 1))
                XX[:, :p] = X_dm
                XX[:, -1] = np.squeeze(Y_dm)
            else:
                XX = np.vstack((X_dm, Y_dm))
            model_cov = NonLinearShrinkageCovariance()
            model_cov.fit(XX)
            cov = model_cov.covariance
            if cov is None:
                raise ValueError("Covariance matrix failed to fit")
            B_11 = cov[:p, :p]
            B_12 = cov[:p, p:]
            B_22 = B_12.T @ la.solve(
                B_11 + self.eps * np.eye(X_dm.shape[1]), B_12
            )
        elif regularization_options["method"] == "LinearShrinkage":
            if q == 1:
                XX = np.zeros((N, p + 1))
                XX[:, :p] = X_dm
                XX[:, -1] = np.squeeze(Y_dm)
            else:
                XX = np.vstack((X_dm, Y_dm))
            model_cov = LinearShrinkageCovariance()
            model_cov.fit(XX)
            cov = model_cov.covariance
            if cov is None:
                raise ValueError("Covariance matrix failed to fit")
            B_11 = cov[:p, :p]
            B_12 = cov[:p, p:]
            B_22 = B_12.T @ la.solve(
                B_11 + self.eps * np.eye(X_dm.shape[1]), B_12
            )
        else:
            raise ValueError(
                "Unrecognized regularization option %s"
                % regularization_options["method"]
            )

        B = np.zeros((p + q, p + q))
        B[:p, :p] = B_11
        B[:p, p:] = B_12
        B[p:, :p] = -self.mu * B_12.T
        B[p:, p:] = -self.mu * B_22

        eigvals, eigvecs = la.eig(B)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = eigvals.argsort()[::-1]
        self.eigvals_ = eigvals[idx]
        V = eigvecs[:, idx]
        W = V[:p, : self.n_components]
        A = W.T @ W
        B = W.T @ X_dm.T
        S = la.solve(A, B).T
        X_recon = S @ W.T
        var = np.mean((X - X_recon) ** 2)
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

    def _fill_regularization_options(self, regularization_options):
        default_options = {
            "method": "LinearShrinkage",
            "prior_options": {"iw_params": {"pnu": 2, "sigma": 3}},
        }
        rops = {**default_options, **regularization_options}
        return rops
