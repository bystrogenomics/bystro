"""
Description:
This file contains implementations related to the PPCASVAE model, a variant 
of Probabilistic Principal Component Analysis (PPCA) tailored for semi-
supervised variational autoencoder structures. It includes the PPCASVAE 
class, which offers functionality for fitting the model to data, 
transforming input features, and reconstructing inputs from latent 
representations. The file also contains utility functions and classes for 
model comparison and evaluation, showcasing the PPCASVAE model's performance 
against traditional PCA and other baseline models in generative feature 
comparison tasks.

Classes:
- PPCASVAE: Implementation of the PPCASVAE model, including methods for 
  fitting to data, encoding, decoding, and utility functions
  for parameter initialization and optimization.

Functions:
- Other utility functions for model comparison and evaluation, including metric 
  computation and visualization tools.

Usage:
This module is intended to be used in machine learning pipelines requiring 
dimensionality reduction, feature extraction, or generative modeling, especially 
in contexts where semi-supervised learning is beneficial. The PPCASVAE model 
can be applied to a wide range of datasets, including but not limited to image 
data, signal data, and general high-dimensional numerical datasets.

Requirements:
- PyTorch: For model implementation and operations.
- NumPy, SciPy: For numerical operations.
- Scikit-learn: For baseline models and preprocessing.
- Matplotlib, Seaborn (optional): For visualization.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Any

from tqdm import trange
import torch
from torch import Tensor, nn
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.linear_model import LogisticRegression, Ridge
import sklearn.decomposition as dp

from bystro.supervised_ppca.gf_generative_pt import PPCA
from bystro.supervised_ppca._base import (
    _get_projection_matrix,
    kl_divergence_vae,
)
from bystro.supervised_ppca._misc_np import softplus_inverse_np


def kl_divergence_gaussian(
    mu0: torch.Tensor,
    sigma0: torch.Tensor,
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence between two multivariate Gaussian distributions.

    Parameters
    ----------
    mu0 : torch.Tensor
        Mean of the first Gaussian distribution, shape (d,).
    sigma0 : torch.Tensor
        Covariance matrix of the first Gaussian distribution, shape (d, d).
    mu1 : torch.Tensor
        Mean of the second Gaussian distribution, shape (d,).
    sigma1 : torch.Tensor
        Covariance matrix of the second Gaussian distribution, shape (d, d).

    Returns
    -------
    torch.Tensor
        The KL divergence between the two Gaussian distributions.
    """
    d = mu0.size(0)
    sigma1_inv = torch.linalg.inv(sigma1)
    trace_term = torch.trace(sigma1_inv @ sigma0)
    mu_diff = mu1 - mu0
    quadratic_term = mu_diff.T @ sigma1_inv @ mu_diff
    log_det_term = torch.logdet(sigma1) - torch.logdet(sigma0)

    kl_div = 0.5 * (log_det_term - d + trace_term + quadratic_term)
    return kl_div


class PPCADropoutVAE(PPCA):
    def __init__(
        self,
        n_components: int = 2,
        n_supervised: int = 1,
        prior_options: dict | None = None,
        mu: float = 1.0,
        gamma: float = 10.0,
        delta: float = 5.0,
        training_options: dict | None = None,
    ):
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.n_supervised = int(n_supervised)
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()
        n_iterations = self.training_options["n_iterations"]
        self.losses_supervision = np.empty(n_iterations)
        self.losses_total = np.empty(n_iterations)
        self.Phi_: NDArray[np.float_] | None = None

    # override needed for mypy to ignore the non-optional `y` argument
    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float_],
        y: NDArray[np.float_],
        task: str = "classification",
        progress_bar: bool = True,
        seed: int = 2021,
    ) -> "PPCADropoutVAE":
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        y : NDArray,(n_samples,n_prediction)
            Covariates we wish to predict. For now lazy and assuming
            logistic regression.

        task : string,default='classification'
            Is this prediction, multinomial regression, or classification

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        seed : int,default=2021
            The random number generator seed used to ensure reproducibility

        Returns
        -------
        self : PPCADropoutVAE
            The model
        """
        self._test_inputs(X, y)
        rng = np.random.default_rng(int(seed))
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        W_, sigmal_ = self._initialize_variables(device, X)
        X_, y_ = self._transform_training_data(device, X, 1.0 * y)

        if task == "classification":
            sigm = nn.Sigmoid()
            supervision_loss: nn.BCELoss | nn.MSELoss = nn.BCELoss(
                reduction="mean"
            )

            mod = LogisticRegression(max_iter=1000)
            mod.fit(X, 1.0 * y)
            b_ = torch.tensor(mod.intercept_.astype(np.float32), device=device)
        elif task == "regression":
            supervision_loss = nn.MSELoss()
        else:
            err_msg = f"unrecognized_task {task}, must be regression or classification"
            raise ValueError(err_msg)

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        one_s = torch.tensor(
            np.ones(self.n_supervised).astype(np.float32), device=device
        )
        softplus = nn.Softplus()

        _prior = self._create_prior(device)

        for i in trange(
            int(training_options["n_iterations"]), disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]
            y_batch = y_[idx]

            sigma = softplus(sigmal_)
            like_prior = _prior(trainable_variables)

            # Predictive lower bound
            P_x, Cov = _get_projection_matrix(W_, sigma, device)
            mean_z = torch.matmul(X_batch, torch.transpose(P_x, 0, 1))
            eps = torch.rand_like(mean_z)
            C1_2 = torch.linalg.cholesky(Cov)
            z_samples = mean_z + torch.matmul(eps, C1_2)

            if task == "regression":
                y_hat = torch.matmul(z_samples[:, : self.n_supervised], one_s)
                loss_y = supervision_loss(y_hat, y_batch)
            else:
                y_hat = (
                    self.delta
                    * torch.matmul(z_samples[:, : self.n_supervised], one_s)
                    + b_
                )
                loss_y = supervision_loss(sigm(y_hat), y_batch)

            # Generative likelihood
            X_recon = torch.matmul(z_samples, W_)
            X_diff = X_batch - X_recon
            m = MultivariateNormal(torch.zeros(p, device=device), sigma * eye)
            like_gen_recon = torch.mean(m.log_prob(X_diff))

            like_gen_kl = torch.mean(kl_divergence_vae(mean_z, Cov))
            like_gen = like_gen_recon - like_gen_kl

            WTW = torch.matmul(W_, torch.transpose(W_, 0, 1))
            off_diag = WTW - torch.diag(torch.diag(WTW))
            loss_i = torch.linalg.matrix_norm(off_diag)

            posterior = like_gen + 1 / N * like_prior
            loss = -1 * posterior + self.mu * loss_y + self.gamma * loss_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_gen, like_prior, posterior)
            if device.type == "cuda":
                self.losses_supervision[i] = loss_y.detach().cpu().numpy()
                self.losses_total[i] = loss.detach().cpu().numpy()
            else:
                self.losses_supervision[i] = loss_y.detach().numpy()
                self.losses_total[i] = loss.detach().numpy()

        self._store_instance_variables(device, trainable_variables)

        if device.type == "cuda":
            self.B_ = b_.detach().cpu().numpy()
        else:
            self.B_ = b_.detach().numpy()

        return self

    def _store_instance_variables(  # type: ignore[override]
        self,
        device: Any,
        trainable_variables: list[Tensor],
    ) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
        device ; pytorch.device
            The device used for trainging (gpu or cpu)

        trainable_variables : list
            List of saved variables of type Tensor

        Sets
        ----
        W_ : NDArray,(n_components,p)
            The loadings

        sigma2_ : float
            The isotropic variance

        """
        if device.type == "cuda":
            self.W_ = trainable_variables[0].detach().cpu().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().cpu().numpy()
            )
        else:
            self.W_ = trainable_variables[0].detach().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().numpy()
            )

    def _test_inputs(self, X, y):
        """
        Just tests to make sure data is numpy array and dimensions match
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data must be numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
        if X.shape[0] != len(y):
            err_msg = "Length of data matrix X must equal length of labels y"
            raise ValueError(err_msg)


class PPCASVAE(PPCA):
    def __init__(
        self,
        n_components: int = 2,
        n_supervised: int = 1,
        prior_options: dict | None = None,
        mu: float = 1.0,
        gamma: float = 10.0,
        delta: float = 5.0,
        lamb: float = 1.0,
        training_options: dict | None = None,
    ):
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.n_supervised = int(n_supervised)
        self.lamb = float(lamb)
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        n_iterations = self.training_options["n_iterations"]
        self._initialize_save_losses()
        self.losses_supervision = np.empty(n_iterations)
        self.losses_encoder = np.empty(n_iterations)
        self.losses_total = np.empty(n_iterations)

    # override needed for mypy to ignore the non-optional `y` argument
    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float_],
        y: NDArray[np.float_],
        task: str = "classification",
        progress_bar: bool = True,
        seed: int = 2021,
    ) -> "PPCASVAE":
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        y : NDArray,(n_samples,n_prediction)
            Covariates we wish to predict. For now lazy and assuming
            logistic regression.

        task : string,default='classification'
            Is this prediction, multinomial regression, or classification

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        seed : int,default=2021
            The random number generator seed used to ensure reproducibility

        Returns
        -------
        self : PPCADropoutVAE
            The model
        """
        self._test_inputs(X, y)
        rng = np.random.default_rng(int(seed))
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        W_, sigmal_, Phi_, sigma_pl_ = self._initialize_variables(device, X)
        X_, y_ = self._transform_training_data(device, X, 1.0 * y)

        if task == "classification":
            sigm = nn.Sigmoid()
            supervision_loss: nn.BCELoss | nn.MSELoss = nn.BCELoss(
                reduction="mean"
            )

            mod = LogisticRegression(max_iter=1000)
            mod.fit(X, 1.0 * y)
            b_ = torch.tensor(mod.intercept_.astype(np.float32), device=device)
        elif task == "regression":
            supervision_loss = nn.MSELoss()
        else:
            err_msg = f"unrecognized_task {task}, must be regression or classification"
            raise ValueError(err_msg)

        trainable_variables = [W_, sigmal_, Phi_, sigma_pl_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        eye_L = torch.tensor(
            np.eye(self.n_components).astype(np.float32), device=device
        )
        one_s = torch.tensor(
            np.ones(self.n_supervised).astype(np.float32), device=device
        )
        softplus = nn.Softplus()

        _prior = self._create_prior(device)

        for i in trange(
            int(training_options["n_iterations"]), disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]
            y_batch = y_[idx]

            sigma = softplus(sigmal_)
            sigma_p = softplus(sigma_pl_) + 0.001

            like_prior = _prior(trainable_variables[:2])

            # Predictive lower bound
            Cov = sigma_p * eye_L
            mean_z = torch.matmul(X_batch, Phi_)
            eps = torch.rand_like(mean_z)
            C1_2 = torch.linalg.cholesky(Cov)
            z_samples = mean_z + torch.matmul(eps, C1_2)

            if task == "regression":
                y_hat = torch.matmul(z_samples[:, : self.n_supervised], one_s)
                loss_y = supervision_loss(y_hat, y_batch)
            else:
                y_hat = (
                    self.delta
                    * torch.matmul(z_samples[:, : self.n_supervised], one_s)
                    + b_
                )
                loss_y = supervision_loss(sigm(y_hat), y_batch)

            # Generative likelihood
            X_recon = torch.matmul(z_samples, W_)
            X_diff = X_batch - X_recon
            m = MultivariateNormal(torch.zeros(p, device=device), sigma * eye)
            like_gen_recon = torch.mean(m.log_prob(X_diff))

            like_gen_kl = torch.mean(kl_divergence_vae(mean_z, Cov))
            like_gen = like_gen_recon - like_gen_kl

            WTW = torch.matmul(W_, torch.transpose(W_, 0, 1))
            off_diag = WTW - torch.diag(torch.diag(WTW))
            loss_i = torch.linalg.matrix_norm(off_diag)

            loss_encoder = torch.mean(torch.square(Phi_)) + torch.mean(
                torch.abs(Phi_)
            )
            loss_decoder = torch.mean(torch.square(W_)) + torch.mean(
                torch.abs(W_)
            )

            posterior = like_gen + 1 / N * like_prior
            loss = (
                -1 * posterior
                + self.mu * loss_y
                + self.gamma * loss_i
                + self.lamb * loss_encoder
                + self.lamb * loss_decoder
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_gen, like_prior, posterior)
            if device.type == "cuda":
                self.losses_supervision[i] = loss_y.detach().cpu().numpy()
                self.losses_encoder[i] = loss_encoder.detach().cpu().numpy()
                self.losses_total[i] = loss.detach().cpu().numpy()

            else:
                self.losses_supervision[i] = loss_y.detach().numpy()
                self.losses_encoder[i] = loss_encoder.detach().numpy()
                self.losses_total[i] = loss.detach().numpy()

        self._store_instance_variables(device, trainable_variables)

        return self

    def _store_instance_variables(  # type: ignore[override]
        self,
        device: Any,
        trainable_variables: list[Tensor],
    ) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
        device ; pytorch.device
            The device used for trainging (gpu or cpu)

        trainable_variables : list
            List of saved variables of type Tensor

        Sets
        ----
        W_ : NDArray,(n_components,p)
            The loadings

        sigma2_ : float
            The isotropic variance

        """
        if device.type == "cuda":
            self.W_ = trainable_variables[0].detach().cpu().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().cpu().numpy()
            )
            self.Phi_ = trainable_variables[2].detach().cpu().numpy()
            self.sigma_enc = (
                nn.Softplus()(trainable_variables[3]).detach().cpu().numpy()
            ) + 0.001
        else:
            self.W_ = trainable_variables[0].detach().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().numpy()
            )
            self.Phi_ = trainable_variables[2].detach().numpy()
            self.sigma2_enc = (
                nn.Softplus()(trainable_variables[3]).detach().numpy()
            ) + 0.001

    def transform_encoder(self, X: NDArray[np.float_]):
        """
        This returns the latent variables as estimated by the encoder,
        as opposed to the generative features

        Parameters
        ----------
        X : np array-like,(N_samples,p)
            The data to transform.

        Returns
        -------
        S : NDArray,(N_samples,n_components)
            The factor estimates
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        mean_z = np.dot(X, self.Phi_)
        return mean_z

    def _initialize_variables(  # type: ignore[override]
        self, device: Any, X: NDArray[np.float_]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Initializes the variables of the model. Right now fits a PCA model
        in sklearn, uses the loadings and sets sigma^2 to be unexplained
        variance.

        Parameters
        ----------
        device ; pytorch.device
            The device used for trainging (gpu or cpu)

        X : NDArray,(n_samples,p)
            The data

        Returns
        -------
        W_ : torch.tensor,shape=(n_components,p)
            The loadings of our latent factor model

        sigmal_ : torch.tensor
            The unrectified variance of the model
        """
        model = dp.PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True, device=device)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = torch.tensor(sinv, requires_grad=True, device=device)
        sigma_pl_ = torch.tensor(sinv, requires_grad=True, device=device)

        model_lm = Ridge(fit_intercept=False)
        model_lm.fit(X, S_hat)
        Phi_ = torch.tensor(
            model_lm.coef_.T.astype(np.float32),
            requires_grad=True,
            device=device,
        )

        return W_, sigmal_, Phi_, sigma_pl_

    def _test_inputs(self, X, y):
        """
        Just tests to make sure data is numpy array and dimensions match
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data must be numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
        if X.shape[0] != len(y):
            err_msg = "Length of data matrix X must equal length of labels y"
            raise ValueError(err_msg)
