import numpy as np
import numpy.linalg as la

from bystro.supervised_ppca._base import BaseGaussianFactorModel
from numpy.typing import NDArray


class PPCAadversarial(BaseGaussianFactorModel):
    def __init__(self, n_components: int = 2, mu=1.0, eps=1e-8):
        super().__init__(n_components=n_components)
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
        self.mean_x = np.mean(X, axis=0)
        self.mean_y = np.mean(Y, axis=0)
        X_dm = X - self.mean_x
        Y_dm = Y - self.mean_y
        N, self.p = X.shape
        p = self.p

        B_11 = X_dm.T @ X_dm
        B_12 = X_dm.T @ Y_dm
        diag_reg = self.eps * np.eye(X_dm.shape[1])

        XtX = X_dm.T @ X_dm + diag_reg
        B_22 = B_12.T @ la.solve(XtX, B_12)
        B = np.concatenate(
            (
                np.concatenate((B_11, -self.mu * B_12), axis=1),
                np.concatenate((B_12, -self.mu * B_22), axis=1),
            ),
            axis=0,
        )
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
