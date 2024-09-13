from typing import Union
import numpy as np
import numpy.linalg as la

from bystro.supervised_ppca._base import BaseGaussianFactorModel
from numpy.typing import NDArray

from sklearn.utils.extmath import randomized_range_finder

from bystro.covariance._covariance_np import (
    EmpiricalCovariance,
    LinearShrinkageCovariance,
    NonLinearShrinkageCovariance,
)
from bystro.covariance.covariance_cov_shrinkage import (
    GeometricInverseShrinkage,
    LinearInverseShrinkage,
    QuadraticInverseShrinkage,
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
        regularization="Linear",
    ):
        super().__init__(n_components=n_components)
        self.regularization = regularization
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
        q = Y.shape[1]

        model_cov = _select_covariance_estimator(self.regularization)
        XX = np.zeros((N, p + q))
        XX[:, :p] = X_dm
        if q == 1:
            XX[:, -1] = np.squeeze(Y_dm)
        else:
            XX[:, p:] = Y_dm
        model_cov.fit(XX)
        cov = model_cov.covariance
        if cov is None:
            raise ValueError("Covariance matrix failed to fit")
        B_11 = cov[:p, :p]
        B_12 = cov[:p, p:]
        B_22 = B_12.T @ la.solve(B_11 + self.eps * np.eye(p), B_12)

        B = np.zeros((p + q, p + q))
        B[:p, :p] = B_11
        B[:p, p:] = B_12
        B[p:, :p] = -self.mu * B_12.T
        B[p:, p:] = -self.mu * B_22
        B = B.T

        eigvals, eigvecs = la.eig(B)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = eigvals.argsort()[::-1]
        self.eigvals_ = eigvals[idx]
        V = eigvecs[:, idx]
        W = V[:p, : self.n_components]
        D = V[p:, : self.n_components]
        A = W.T @ W
        B = W.T @ X_dm.T
        S = la.solve(A, B).T
        X_recon = S @ W.T
        var = np.mean((X - X_recon) ** 2)
        self._store_instance_variables((W, var, D))

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
        self,
        trainable_variables: tuple[
            NDArray[np.float_], np.float_, NDArray[np.float_]
        ],
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
        self.D_ = trainable_variables[2].T

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


class PPCAsupervised(BaseGaussianFactorModel):
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
        The supervision strength which controls the penalty for
        failing to predict the supervised variables in the latent
        representation.

    eps : float
        A small constant added to improve the conditioning of
        matrix inverses during computation.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

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
        regularization="Linear",
    ):
        super().__init__(n_components=n_components)
        self.regularization = regularization
        self.mu: float = mu
        self.eps: float = eps
        self.W_: NDArray[np.float_] | None = None
        self.Phi_: NDArray[np.float_] | None = None
        self.p: int | None = None
        self.sigma2_: np.float_ | None = None

    def fit(
        self, X: NDArray[np.float_], Y: NDArray[np.float_]
    ) -> "PPCAsupervised":
        """
        Fits a model given covariates X

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        Y : NDArray,(n_samples,n_predicted)
            The data we want to predict

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
        q = Y.shape[1]

        model_cov = _select_covariance_estimator(self.regularization)
        XX = np.zeros((N, p + q))
        XX[:, :p] = X_dm
        if q == 1:
            XX[:, -1] = np.squeeze(Y_dm)
        else:
            XX[:, p:] = Y_dm
        model_cov.fit(XX)
        cov = model_cov.covariance
        if cov is None:
            raise ValueError("Covariance matrix failed to fit")
        B_11 = cov[:p, :p]
        B_12 = cov[:p, p:]
        B_22 = B_12.T @ la.solve(B_11 + self.eps * np.eye(p), B_12)

        B = np.zeros((p + q, p + q))
        B[:p, :p] = B_11
        B[:p, p:] = B_12
        B[p:, :p] = self.mu * B_12.T
        B[p:, p:] = self.mu * B_22
        B = B.T

        eigvals, eigvecs = la.eig(B)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = eigvals.argsort()[::-1]
        self.eigvals_ = eigvals[idx]
        V = eigvecs[:, idx]
        W = V[:p, : self.n_components]
        D = V[p:, : self.n_components]
        A = W.T @ W
        B = W.T @ X_dm.T
        S = la.solve(A, B).T
        X_recon = S @ W.T
        var = np.mean((X - X_recon) ** 2)
        self._store_instance_variables((W, var, D))

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
        self,
        trainable_variables: tuple[
            NDArray[np.float_], np.float_, NDArray[np.float_]
        ],
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
        self.Phi_ = trainable_variables[2].T

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


class PPCASupAdversarial(BaseGaussianFactorModel):
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
        The supervision strength which controls the penalty for
        predicting the predicted variables in the latent representation.

    eps : float
        A small constant added to improve the conditioning of
        matrix inverses during computation.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    Notes
    -----
    This implementation is based on modifications to the
    'augmented-pca' package by Carson, Talbot and Carlson
    """

    def __init__(
        self,
        n_components: int = 2,
        mu_s=1.0,
        mu_a=1.0,
        eps=1e-8,
        regularization="Linear",
    ):
        super().__init__(n_components=n_components)
        self.regularization = regularization
        self.mu_s: float = mu_s
        self.mu_a: float = mu_a
        self.eps: float = eps
        self.W_: NDArray[np.float_] | None = None
        self.Phi_: NDArray[np.float_] | None = None
        self.D_: NDArray[np.float_] | None = None
        self.p: int | None = None
        self.sigma2_: np.float_ | None = None

    def fit(
        self,
        X: NDArray[np.float_],
        Y: NDArray[np.float_],
        Z: NDArray[np.float_],
    ) -> "PPCASupAdversarial":
        """
        Fits a model given covariates X

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        Y : NDArray,(n_samples,n_predictors)
            The data we want to predict

        Z : NDArray,(n_samples,n_confounders)
            The concommitant data we want to remove

        Returns
        -------
        self : PPCA_sup_adversarial
            The model
        """
        self.mean_x = np.mean(X, axis=0)
        self.mean_y = np.mean(Y, axis=0)
        self.mean_z = np.mean(Z, axis=0)
        X_dm = X - self.mean_x
        Y_dm = Y - self.mean_y
        Z_dm = Z - self.mean_z
        N, self.p = X.shape
        p = self.p
        q1 = Y.shape[1]
        q2 = Z.shape[1]

        model_cov = _select_covariance_estimator(self.regularization)
        XX = np.zeros((N, p + q1 + q2))
        XX[:, :p] = X_dm

        if q1 == 1:
            XX[:, p] = np.squeeze(Y_dm)
        else:
            XX[:, p : (p + q1)] = Y_dm

        if q2 == 1:
            XX[:, -1] = np.squeeze(Z_dm)
        else:
            XX[:, (p + q1) :] = Z_dm

        model_cov.fit(XX)
        cov = model_cov.covariance
        if cov is None:
            raise ValueError("Covariance matrix failed to fit")
        B_11 = cov[:p, :p]
        B_12 = cov[:p, p : (p + q1)]
        B_13 = cov[:p, (p + q1) :]

        B_22 = B_12.T @ la.solve(B_11 + self.eps * np.eye(p), B_12)
        B_33 = B_13.T @ la.solve(B_11 + self.eps * np.eye(p), B_13)

        B_23 = B_12.T @ la.solve(B_11 + self.eps * np.eye(p), B_13)

        B = np.zeros((p + q1 + q2, p + q1 + q2))
        B[:p, :p] = B_11
        B[:p, p : (p + q1)] = B_12
        B[:p, (p + q1) :] = B_13

        B[p : (p + q1), :p] = self.mu_s * B_12.T
        B[p : (p + q1), p : (p + q1)] = self.mu_s * B_22
        B[p : (p + q1), (p + q1) :] = self.mu_s * B_23

        B[(p + q1 + q2) :, :p] = -self.mu_a * B_13.T
        B[(p + q1 + q2) :, p : (p + q1)] = -self.mu_a * B_23.T
        B[(p + q1 + q2) :, (p + q1) :] = -self.mu_a * B_33

        eigvals, eigvecs = la.eig(B)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = eigvals.argsort()[::-1]
        self.eigvals_ = eigvals[idx]
        V = eigvecs[:, idx]
        W = V[:p, : self.n_components]
        D1 = V[p : (p + q1), : self.n_components]
        D2 = V[(p + q1) :, : self.n_components]
        A = W.T @ W
        B = W.T @ X_dm.T
        S = la.solve(A, B).T
        X_recon = S @ W.T
        var = np.mean((X - X_recon) ** 2)
        self._store_instance_variables((W, var, D1, D2))

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
        self,
        trainable_variables: tuple[
            NDArray[np.float_],
            np.float_,
            NDArray[np.float_],
            NDArray[np.float_],
        ],
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
        self.D_ = trainable_variables[2].T

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


def _select_covariance_estimator(regularization):
    """
    This is a tiny method for selecting the estimator for the covariance
    matrix.
    """
    model_cov: Union[
        EmpiricalCovariance,
        LinearShrinkageCovariance,
        NonLinearShrinkageCovariance,
        LinearInverseShrinkage,
        GeometricInverseShrinkage,
        QuadraticInverseShrinkage,
    ]
    if regularization == "Empirical":
        model_cov = EmpiricalCovariance()
    elif regularization == "Linear":
        model_cov = LinearShrinkageCovariance()
    elif regularization == "LinearInverse":
        model_cov = LinearInverseShrinkage()
    elif regularization == "QuadraticInverse":
        model_cov = QuadraticInverseShrinkage()
    elif regularization == "GeometricInverse":
        model_cov = GeometricInverseShrinkage()
    elif regularization == "NonLinear":
        model_cov = NonLinearShrinkageCovariance()
    elif regularization == "Bayesian":
        raise ValueError("Bayesian currently not supported")
    else:
        raise ValueError("Unrecognized regularization %s" % regularization)
    return model_cov


class PPCAadversarialRandomized(BaseGaussianFactorModel):
    def __init__(
        self,
        n_components: int = 2,
        mu: float = 1.0,
        eps: float = 1e-8,
        n_oversamples: int = 20,
        random_state: int = 2021,
    ):
        super().__init__(n_components=n_components)
        self.mu: float = mu
        self.eps: float = eps
        self.W_: NDArray[np.float_] | None = None
        self.D_: NDArray[np.float_] | None = None
        self.p: int | None = None
        self.n_oversamples: int = n_oversamples
        self.random_state: int = random_state
        self.sigma2_: np.float_ | None = None

    def fit(
        self, X: NDArray[np.float_], Y: NDArray[np.float_]
    ) -> "PPCAadversarialRandomized":
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
        q = Y.shape[1]

        n_random = self.n_components + self.n_oversamples
        Q = randomized_range_finder(
            X_dm.T,
            n_iter=7,
            size=n_random,
            power_iteration_normalizer="auto",
            random_state=self.random_state,
        )

        X_tilde = Q.T @ X_dm.T

        B_11 = np.cov(X_tilde)
        B_12 = 1 / N * np.dot(X_tilde, Y_dm)
        B_22 = B_12.T @ la.solve(B_11 + self.eps * np.eye(n_random), B_12)

        B = np.zeros((n_random + q, n_random + q))
        B[:n_random, :n_random] = B_11
        B[:n_random, n_random:] = B_12
        B[n_random:, :n_random] = -self.mu * B_12.T
        B[n_random:, n_random:] = -self.mu * B_22
        B = B.T

        eigvals, eigvecs = la.eig(B)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = eigvals.argsort()[::-1]

        self.eigvals_ = eigvals[idx]
        V_latent = eigvecs[:, idx]

        self.V_latent = V_latent

        W_latent = V_latent[:n_random, : self.n_components]

        W = Q @ W_latent

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
        self,
        trainable_variables: tuple[
            NDArray[np.float_], np.float_, 
        ],
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


"""
https://github.com/wecarsoniv/augmented-pca/tree/main

Copyright (c) 2021 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
