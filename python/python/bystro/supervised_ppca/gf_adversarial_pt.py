"""
This implements probabilistic pca but with an adversary to ensure that the 
latent space is unpredictive of a sensitive variable. This corresponds to 
maximizing the log likelihood with a constraint on the mutual information 
between the generative model and the sensitive variable.

Note that in the code W is a L x p array, where L is the latent dimensionality
and  p is the covariate dimension, for implementation convenience. However,
mathematically it is often pxL for notational convenience. Given that the
most insidious errors are mathematical in nature rather than coding, (as faulty
math is difficult to detect in unit tests), our notation matches mathematics
rather than code, specifically when naming WWT and WTW to match the Bishop 2006
notation rather than code.


Objects
-------
PPCAAdversarial(PPCA)
    This is PPCA with SGD but removes influence of sensitive variable.
Methods
-------
None
"""
import numpy as np
from numpy.typing import NDArray

from tqdm import trange
import torch
from torch import Tensor, nn
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.decomposition import PCA

from bystro.supervised_ppca._misc_np import softplus_inverse_np
from bystro.supervised_ppca.gf_generative_pt import PPCA


def _get_projection_matrix(W_: Tensor, sigma_: Tensor):
    """
    This is currently just implemented for PPCA due to nicer formula. Will
    modify for broader application later.

    Computes the parameters for p(S|X)

    Description in future to be released paper

    Parameters
    ----------
    W_ : Tensor(n_components,p)
        The loadings

    sigma_ : Tensor
        Isotropic noise

    Returns
    -------
    Proj_X : Tensor(n_components,p)
        Beta such that np.dot(Proj_X, X) = E[S|X]

    Cov : Tensor(n_components,n_components)
        Var(S|X)
    """
    n_components = int(W_.shape[0])
    eye = torch.tensor(np.eye(n_components), dtype=torch.float32)
    M = torch.matmul(W_, torch.transpose(W_, 0, 1)) + sigma_ * eye
    Proj_X = torch.linalg.solve(M, W_)
    Cov = torch.linalg.inv(M) * sigma_
    return Proj_X, Cov


class PPCAAdversarial(PPCA):
    """
    This is an adversarial form of PCA, which seeks to maximize the
    objective

    max_{W,sigma2} log N(X;0,sigma2I + WW^T)
    s.t. MI(XW,Y) = 0

    where Y is our confounding variable.

    Parameters
    ----------
    n_components : int
        The latent dimensionality

    prior_options : dict
        The hyperparameters for the prior on the parameters

    mu : float>0,default=1.0
        The adversarial penalty

    gamma : float,default=10.0
        Ensures that the components are orthogonal

    training_options : dict
        The options used for stochastic gradient descent for adversary
        and generative model
    """

    def __init__(
        self,
        n_components: int = 2,
        prior_options: dict | None = None,
        mu: float = 1.0,
        gamma: float = 1.0,
        training_options: dict | None = None,
    ):
        self.mu = float(mu)
        self.gamma = float(gamma)
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()
        self.losses_supervision = np.empty(
            self.training_options["n_iterations"]
        )

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float_],
        y: NDArray[np.float_],
        task: str = "classification",
        progress_bar: bool = True,
        seed: int = 2021,
    ) -> "PPCAAdversarial":
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        y : NDArray,(n_samples,)
            Covariates we wish to make orthogonal to the latent space

        task : string,default='classification'
            Is this prediction or classification

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        seed : int,default=2021
            The random number generator seed used to ensure reproducibility

        Returns
        -------
        self : PPCAAdversarial
            The model
        """
        self._test_inputs(X, y)
        rng = np.random.default_rng(int(seed))
        training_options = self.training_options
        N, p = X.shape
        self.p = p

        discriminator = nn.Sequential()
        discriminator.add_module("dense1", nn.Linear(self.n_components, 30))
        discriminator.add_module("act1", nn.ReLU())
        discriminator.add_module("dense2", nn.Linear(30, 1))

        W_, sigmal_ = self._initialize_variables(X)
        X_, y_ = self._transform_training_data(X, 1.0 * y)
        X_ = X_.type(torch.float32)
        y_ = y_.type(torch.float32)

        if task == "classification":
            sigm = nn.Sigmoid()
            supervision_loss: nn.BCELoss | nn.MSELoss = nn.BCELoss()
        elif task == "regression":
            supervision_loss = nn.MSELoss()
        else:
            err_msg = f"unrecognized_task {task}, must be regression or classification"
            raise ValueError(err_msg)

        trainable_variables = [W_, sigmal_]

        optimizer_g = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        optimizer_d = torch.optim.SGD(
            discriminator.parameters(),
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        eye = torch.tensor(np.eye(p), dtype=torch.float32)
        zerosp = torch.zeros(p, dtype=torch.float32)
        softplus = nn.Softplus()

        _prior = self._create_prior()

        for i in trange(
            int(training_options["n_iterations"]), disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]
            y_batch = y_[idx]

            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            like_prior = _prior(trainable_variables)

            m = MultivariateNormal(zerosp, Sigma)
            like_gen = torch.mean(m.log_prob(X_batch))

            P_x, Cov = _get_projection_matrix(W_, sigma)
            mean_z = torch.matmul(X_batch, torch.transpose(P_x, 0, 1))
            eps = torch.rand_like(mean_z)
            C1_2 = torch.linalg.cholesky(Cov)
            z_samples = mean_z + torch.matmul(eps, C1_2)

            y_hat = torch.squeeze(discriminator(z_samples))

            if task == "regression":
                loss_y = supervision_loss(y_hat, y_batch)
            else:
                loss_y = supervision_loss(sigm(y_hat), y_batch)

            optimizer_d.zero_grad()
            loss_y.backward(retain_graph=True)
            optimizer_d.step()

            y_hat2 = torch.squeeze(discriminator(z_samples))
            if task == "regression":
                loss_y2 = supervision_loss(y_hat2, y_batch)
            else:
                loss_y2 = supervision_loss(sigm(y_hat2), y_batch)

            WTW = torch.matmul(W_, torch.transpose(W_, 0, 1))
            off_diag = WTW - torch.diag(torch.diag(WTW))
            loss_i = torch.linalg.matrix_norm(off_diag)

            posterior = like_gen + 1 / N * like_prior
            loss_gen = -1 * posterior - self.mu * loss_y2 + self.gamma * loss_i

            optimizer_g.zero_grad()
            loss_gen.backward()
            optimizer_g.step()

            self._save_losses(i, like_gen, like_prior, posterior)
            self.losses_supervision[i] = loss_y.detach().numpy()

        self._store_instance_variables(trainable_variables)

        return self

    def _store_instance_variables(self, trainable_variables: list[Tensor]):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of saved variables of type Tensor

        Sets
        ----
        W_ : NDArray,(n_components,p)
            The loadings

        sigma2_ : float
            The isotropic variance

        B_ : float
            The intercept for the predictive model
        """
        self.W_ = trainable_variables[0].detach().numpy()
        self.sigma2_ = nn.Softplus()(trainable_variables[1]).detach().numpy()

    def _initialize_variables(self, X: NDArray):
        """
        Initializes the variables of the model. Right now fits a PCA model
        in sklearn, uses the loadings and sets sigma^2 to be unexplained
        variance.

        Parameters
        ----------
        X : NDArray,(n_samples,p)
            The data

        Returns
        -------
        W_ : torch.tensor,shape=(n_components,p)
            The loadings of our latent factor model

        sigmal_ : torch.tensor
            The unrectified variance of the model

        B_ : torch.tensor
            The predictive model intercept y = XW + B_
        """
        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_
        W_ = torch.tensor(W_init, requires_grad=True, dtype=torch.float32)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = torch.tensor(sinv, requires_grad=True, dtype=torch.float32)
        return W_, sigmal_

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
