"""
This implements Factor analysis but with variational inference to allow us
to supervise a subset of latent variables to be predictive of an auxiliary 
outcome. Currently only implements logistic regression but in the future 
will be modified to allow for more general predictive outcomes.

Objects
-------
PPCADropout(PPCA)
    This is PPCA with SGD but supervises a single factor to predict y.

TODO
SPPCADropout(SPCA)
    This is SPCA with SGD but supervises a single factor. 

TODO
FactorAnalysisDropout(FactorAnalysis)
    This is factor analysis but supervises a single factor.

Methods
-------
None
"""
import numpy as np

from tqdm import trange
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.decomposition import PCA  # type: ignore

from bystro.supervised_ppca._misc_np import softplus_inverse_np
from bystro.supervised_ppca.gf_generative_pt import PPCA


def _get_projection_matrix(W_, sigma_):
    """
    This is currently just implemented for PPCA due to nicer formula. Will
    modify for broader application later.

    Computes the parameters for p(S|X)

    Description in future to be released paper

    Parameters
    ----------
    W_ : pt.Tensor(n_components,p)
        The loadings

    sigma_ : pt.tensor
        Isotropic noise

    Returns
    -------
    Proj_X : pt.tensor(n_components,p)
        Beta such that np.dot(Proj_X, X) = E[S|X]

    Cov : pt.tensor(n_components,n_components)
        Var(S|X)
    """
    n_components = int(W_.shape[0])
    eye = torch.tensor(np.eye(n_components))
    M = torch.matmul(W_, torch.transpose(W_, 0, 1)) + sigma_ * eye
    Proj_X = torch.linalg.solve(M, W_)
    Cov = torch.linalg.inv(M) * sigma_
    return Proj_X, Cov


class PPCADropout(PPCA):
    """
    This implements supervised PPCA according to the paper draft that I'm
    working on. That is the generative mechanism matches probabilistic 
    PCA with  isotropic variance. However, a variational lower bound on
    a predictive objective is used to ensure that a subset of latent 
    variables are predictive of an auxiliary task.

    Parameters
    ----------
    n_components : int
        The latent dimensionality

    n_supervised : int
        The number of predictive latent variables
            
    prior_options : dict
        The hyperparameters for the prior on the parameters

    mu : float>0,default=1.0

    gamma : float,default=10.0

    delta : 5.0
        

    """

    def __init__(
        self,
        n_components=2,
        n_supervised=1,
        prior_options=None,
        mu=1.0,
        gamma=10.0,
        delta=5.0,
        training_options=None,
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
        self.losses_supervision = np.empty(
            self.training_options["n_iterations"]
        )
        self.W_ = None
        self.sigma2_ = None
        self.B_ = None

    def fit(self, X, y, task="classification", progress_bar=True, seed=2021):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : np.array-like,(n_samples,n_prediction)
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
        self : object
            The model
        """
        self._test_inputs(X, y)
        rng = np.random.default_rng(int(seed))
        training_options = self.training_options
        N, p = X.shape
        self.p = p

        W_, sigmal_, B_ = self._initialize_variables(X)
        X, y = self._transform_training_data(X, 1.0 * y)

        if task == "classification":
            sigm = nn.Sigmoid()
            supervision_loss = nn.BCELoss()
        elif task == "regression":
            supervision_loss = nn.MSELoss()
        else:
            err_msg = (
                f"unrecognized_task {task}, must be regression or classification"
            )
            raise ValueError(err_msg)

        trainable_variables = [W_, sigmal_, B_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        eye = torch.tensor(np.eye(p))
        one_s = torch.tensor(np.ones(self.n_supervised))
        softplus = nn.Softplus()

        _prior = self._create_prior()

        myrange = trange if progress_bar else range

        for i in myrange(training_options["n_iterations"]):
            idx = rng.choice(
                X.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X[idx]
            y_batch = y[idx]

            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            like_prior = _prior(trainable_variables)

            # Generative likelihood
            m = MultivariateNormal(torch.zeros(p), Sigma)
            like_gen = torch.mean(m.log_prob(X_batch))

            # Predictive lower bound
            P_x, Cov = _get_projection_matrix(W_, sigma)
            mean_z = torch.matmul(X_batch, torch.transpose(P_x, 0, 1))
            eps = torch.rand_like(mean_z)
            C1_2 = torch.linalg.cholesky(Cov)
            z_samples = mean_z + torch.matmul(eps, C1_2)

            y_hat = (
                self.delta
                * torch.matmul(z_samples[:, : self.n_supervised], one_s)
                + B_
            )
            if task == "regression":
                loss_y = supervision_loss(y_hat, y_batch)
            else:
                loss_y = supervision_loss(sigm(y_hat), y_batch)

            WTW = torch.matmul(W_, torch.transpose(W_, 0, 1))
            off_diag = WTW - torch.diag(torch.diag(WTW))
            loss_i = torch.linalg.matrix_norm(off_diag)

            posterior = like_gen + 1 / N * like_prior
            loss = -1 * posterior + self.mu * loss_y + self.gamma*loss_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, like_gen, like_prior, posterior)
            self.losses_supervision[i] = loss_y.detach().numpy()

        self._store_instance_variables(trainable_variables)
        return self

    def _store_instance_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of tensorflow variables saved

        Sets
        ----
        W_ : np.array-like,(n_components,p)
            The loadings

        sigma2_ : float
            The isotropic variance

        B_ : float
            The intercept for the predictive model
        """
        self.W_ = trainable_variables[0].detach().numpy()
        self.sigma2_ = nn.Softplus()(trainable_variables[1]).detach().numpy()
        self.B_ = trainable_variables[2].detach().numpy()

    def _initialize_variables(self, X):
        """
        Initializes the variables of the model. Right now fits a PCA model
        in sklearn, uses the loadings and sets sigma^2 to be unexplained
        variance.

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
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
        W_ = torch.tensor(W_init, requires_grad=True)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = torch.tensor(sinv, requires_grad=True)
        B_ = torch.tensor(0.0, requires_grad=True)
        return W_, sigmal_, B_

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
