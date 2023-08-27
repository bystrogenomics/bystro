"""
This implements Factor analysis but with variational inference to allow us
to supervise a single latent variable to be predictive of an auxiliary 
outcome. Currently only implements logistic regression but in the future 
will be modified to allow for more general predictive outcomes.

Objects
-------
PPCADropoutpt(PPCApt)
    This is PPCA with SGD but supervises a single factor to predict y.

SPPCADropoutpt(SPCApt)
    This is SPCA with SGD but supervises a single factor. Currently 
    unimplemented since I need to get this PR in.

FactorAnalysisDropoutpt(FactorAnalysispt)
    This is factor analysis but supervises a single factor. Currently 
    unimplemented since I need to get this PR in.

Methods
-------
_get_projection_matrix(W_,sigma_,n_components)
    Computes the parameters for p(S|X)

"""
import numpy as np

from sklearn import decomposition as dp
from tqdm import trange
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from ._misc_np import softplus_inverse_np
from .gf_generative_pt import PPCApt


def _get_projection_matrix(W_, sigma_):
    """
    This is currently just implemented for PPCA due to nicer formula. Will
    modify for broader application later.

    Computes the parameters for p(S|X)

    Parameters
    ----------
    W_ : pt.Tensor(n_components,p)
        The loadings

    sigma_ : pt.Flaot
        Isotropic noise

    Returns
    -------
    Proj_X : pt.tensor(n_components,p)
        Beta such that Proj_XX = E[S|X]

    Cov : pt.tensor(n_components,n_components)
        Var(S|X)
    """
    n_components = int(W_.shape[0])
    eye = torch.tensor(np.eye(n_components))  # .astype(np.float32))
    M = torch.matmul(W_, torch.transpose(W_, 0, 1)) + sigma_ * eye
    Proj_X = torch.linalg.solve(M, W_)
    Cov = torch.linalg.inv(M) * sigma_
    return Proj_X, Cov


class PPCADropoutpt(PPCApt):
    def __init__(
        self,
        n_components=2,
        n_supervised=1,
        prior_options={},
        mu=1.0,
        gamma=10.0,
        delta=5.0,
        training_options={},
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

    def fit(self, X, y=None, task="classification"):
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

        Returns
        -------
        self : object
            The model
        """
        rng = np.random.default_rng(2021)
        td = self.training_options
        N, p = X.shape
        self.p = p

        W_, sigmal_ = self._initialize_variables(X)
        X = torch.tensor(X)
        y = torch.tensor(1.0 * y)

        if task == "classification":
            sigm = nn.Sigmoid()
            supervision_loss = nn.BCELoss()
        elif task == "regression":
            supervision_loss = nn.MSELoss()
        else:
            supervision_loss = nn.BCELoss()

        B_ = torch.tensor(0.0, requires_grad=True)
        trainable_variables = [W_, sigmal_, B_]

        self._initialize_saved_losses()
        self.losses_supervision = np.zeros(td["n_iterations"])

        optimizer = torch.optim.SGD(
            trainable_variables, lr=td["learning_rate"], momentum=0.9
        )
        eye = torch.tensor(np.eye(p))
        torch.tensor(np.zeros(self.n_components))
        one_s = torch.tensor(np.ones(self.n_supervised))
        softplus = nn.Softplus()
        torch.tensor(np.zeros(self.n_supervised))

        _prior = self._create_prior()

        for i in trange(td["n_iterations"]):
            idx = rng.choice(X.shape[0], size=td["batch_size"], replace=False)
            X_batch = X[idx]
            y_batch = y[idx]

            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            like_prior = _prior(W_, sigma)

            # Generative likelihood
            m = MultivariateNormal(torch.zeros(p), Sigma)
            like_gen = torch.mean(m.log_prob(X_batch))

            # Predictive lower bound
            P_x, Cov = _get_projection_matrix(W_, sigma)
            mean_z = torch.matmul(X_batch, torch.transpose(P_x, 0, 1))
            eps = torch.rand_like(mean_z)
            C1_2 = torch.cholesky(Cov)
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
            torch.norm(off_diag)

            posterior = like_gen + 1 / N * like_prior
            loss = -1 * posterior + self.mu * loss_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, like_gen, like_prior, posterior)
            self.losses_supervision[i] = loss_y.detach().numpy()

        self._save_variables(trainable_variables)
        return self

    def _save_variables(self, trainable_variables):
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

        sigmas_ : np.array-like,(n_components,p)
            The diagonal variances
        """
        self.W_ = trainable_variables[0].detach().numpy()
        self.sigma2_ = nn.Softplus()(trainable_variables[1]).detach().numpy()
        self.B_ = trainable_variables[2].detach().numpy()

    def _initialize_variables(self, X):
        """
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        Returns
        -------
        W_ : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : tf.Float
            The variance of the model
        """
        model = dp.PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_
        W_ = torch.tensor(W_init, requires_grad=True)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = torch.tensor(sinv, requires_grad=True)
        return W_, sigmal_
