"""
This implements the following model:

    p(z) ~ GMM({mu}_k,I)
    p(x|z) ~ N(Wz,sigma^2I)

This is different for a standard mixture model and from 
probabilistic PCA in that it ties the covariance and means
together in a very specific format. This corresponds to 
fitting a high dimensional GMM such that the latent 
distribution is interpretable. No guarantees on functionality,
just on correctness.

Objects
-------
GaussianMixturePPCA
    The model implementation of a Gaussian mixture model 
    closely corresponding to probabilistic PCA.

Methods
-------
None
"""
import numpy as np
from numpy.typing import NDArray
import numpy.linalg as la

from sklearn.decomposition import PCA  # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from tqdm import trange

from bystro._template_sgd_np import BaseSGDModel  # type: ignore


class GaussianMixturePPCA(BaseSGDModel):
    """
    This fits the following generative model

    p(z) ~ GMM({mu}_k,I)
    p(x|z) ~ N(Wz,sigma^2I)

    using stochastic gradient descent on the
    marginal likelihood.

    """

    def __init__(
        self,
        n_clusters,
        n_components,
        training_options=None,
        prior_options=None,
    ):
        """
        This is a Gaussian mixture model with a shared low
        rank covariance structure.

        Paramters
        ---------
        n_clusters : int
            Number of groups in the latent space

        n_components : int
            PPCA dimensionality
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        if training_options is None:
            training_options = {}
        if prior_options is None:
            prior_options = {}

        self.training_options = self._fill_training_options(training_options)
        self.prior_options = self._fill_prior_options(prior_options)

    def _fill_training_options(self, training_options):
        """
        This sets the default parameters for stochastic gradient descent,
        our inference strategy for the model.

        Parameters
        ----------
        training_options : dict
            The original options set by the user passed as a dictionary

        Options
        -------
        n_iterations : int, default=3000
            Number of iterations to train using stochastic gradient descent

        learning_rate : float, default=1e-4
            Learning rate of gradient descent

        batch_size : int, default=None
            The number of observations to use at each iteration. If none
            corresponds to batch learning
        """
        default_options = {
            "n_iterations": 3000,
            "learning_rate": 1e-3,
            "batch_size": 100,
            "momentum": 0.9,
        }
        tops = {**default_options, **training_options}

        default_keys = set(default_options.keys())
        final_keys = set(tops.keys())

        expected_but_missing_keys = default_keys - final_keys
        unexpected_but_present_keys = final_keys - default_keys
        if expected_but_missing_keys:
            raise ValueError("Missing keys")
        if unexpected_but_present_keys:
            raise ValueError("Extra keys")
        return tops

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters on latent space

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        default_dict = {"mu_l2": 1.0}
        new_dict = {**default_dict, **prior_options}
        return new_dict

    def fit(self, X, progress_bar=True, seed=2021):
        """
        Fits a model given covariates X

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        Returns
        -------
        self : object
            The model
        """
        X = X.astype(np.float32)
        self._test_inputs(X)
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        rng = np.random.default_rng(int(seed))
        K = self.n_clusters

        W_, sigmal_, pi_logits, mu_list = self._initialize_variables(X)

        X = self._transform_training_data(X)[0]

        trainable_variables = [W_, sigmal_, pi_logits] + mu_list

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        softplus = nn.Softplus()
        mse = nn.MSELoss()
        smax = nn.Softmax()

        eye = torch.tensor(np.eye(p).astype(np.float32))

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            idx = rng.choice(
                X.shape[0], size=training_options["batch_size"], replace=False
            )

            X_batch = X[idx]

            sigma2 = softplus(sigmal_)

            X_each = [X_batch - torch.matmul(mu_list[k], W_) for k in range(K)]

            Sigma = torch.matmul(torch.transpose(W_, 0, 1), W_) + sigma2 * eye

            m = MultivariateNormal(torch.zeros(p), Sigma)

            pi_ = smax(pi_logits)
            loss_logits = 0.001 * mse(pi_logits, torch.zeros(K))

            log_likelihood_each = [
                m.log_prob(X_each[k]) for k in range(K)
            ]  # List of batchsize x 1
            log_likelihood_stack = torch.stack(
                log_likelihood_each
            )  # matrix of batchsize x K
            log_likelihood_components = torch.transpose(
                log_likelihood_stack, 0, 1
            ) + torch.log(
                pi_
            )  # Log component posterior
            log_likelihood_marg = torch.logsumexp(
                log_likelihood_components, dim=1
            )  # Log likelihood per component
            loss_likelihood = -1 * torch.mean(
                log_likelihood_marg
            )  # Loss function of likelihood

            loss = loss_logits + loss_likelihood

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._store_instance_variables(trainable_variables)
        return self

    def get_covariance(self):
        """
        Gets the covariance matrix defined by the model parameters

        Parameters
        ----------
        None

        Returns
        -------
        covariance : np.array-like(p,p)
            The covariance matrix
        """
        covariance = np.dot(self.W_.T, self.W_) + np.diag(self.sigma2_)
        return covariance

    def get_precision(self):
        """
        Gets the precision matrix defined as the inverse of the covariance

        Parameters
        ----------
        None

        Returns
        -------
        precision : np.array-like(p,p)
            The inverse of the covariance matrix
        """
        covariance = self.get_covariance()
        precision = la.inv(covariance)
        return precision

    def transform(self, X):
        """
        This returns the latent variable estimates given X

        Parameters
        ----------
        X : np array-like,(N_samples,p
                            The data to transform.

        Returns
        -------
        S : np.array-like,(N_samples,n_components)
            The factor estimates
        """
        prec = self.get_precision()
        coefs = np.dot(self.W_, prec)
        S = np.dot(X, coefs.T)
        return S

    def _initialize_save_losses(self):
        """
        This method initializes the arrays to track relevant variables
        during training at each iteration

        Sets
        ----
        losses_likelihood : np.array(n_iterations)
            The log likelihood

        losses_prior : np.array(n_iterations)
            The log prior

        losses_posterior : np.array(n_iterations)
            The log posterior
        """
        n_iterations = self.training_options["n_iterations"]
        self.losses_likelihood = np.empty(n_iterations)
        self.losses_prior = np.empty(n_iterations)
        self.losses_posterior = np.empty(n_iterations)

    def _initialize_variables(self, X):
        """
        Initializes the variables of the model. Right now fits a PCA model
        in sklearn, uses the loadings and sets sigma^2 to be unexplained
        variance for each group.

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        Returns
        -------
        W_ : torch.tensor-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : list
            A list of the isotropic noises for each group
        """
        model_pca = PCA(self.n_components)
        S_ = model_pca.fit_transform(X)
        model_gmm = GaussianMixture(self.n_clusters, covariance_type="tied")
        model_gmm.fit(S_)
        W_init = model_pca.components_
        W_ = torch.tensor(W_init.astype(np.float32), requires_grad=True)
        X_recon = np.dot(S_, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = torch.tensor(sinv, requires_grad=True)

        pi_logits = torch.tensor(
            np.log(model_gmm.weights_ + 1e-3).astype(np.float32),
            requires_grad=True,
        )
        mean_list = [
            torch.tensor(
                model_gmm.means_[k].astype(np.float32), requires_grad=True
            )
            for k in range(self.n_clusters)
        ]
        return W_, sigmal_, pi_logits, mean_list

    def _save_losses(self, i, log_likelihood, log_prior, log_posterior):
        """
        Saves the values of the losses at each iteration

        Parameters
        -----------
        i : int
            Current training iteration

        losses_likelihood : torch.tensor
            The log likelihood

        losses_prior : torch.tensor
            The log prior

        losses_posterior : torch.tensor
            The log posterior
        """
        self.losses_likelihood[i] = log_likelihood.detach().numpy()
        if isinstance(log_prior, np.ndarray):
            self.losses_prior[i] = log_prior
        else:
            self.losses_prior[i] = log_prior.detach().numpy()
        self.losses_posterior[i] = log_posterior.detach().numpy()

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
        """
        self.W_ = trainable_variables[0].detach().numpy()
        self.sigma2_ = nn.Softplus()(trainable_variables[1]).detach().numpy()
        self.pi_ = nn.Softmax()(trainable_variables[2]).detach().numpy()
        self.mu_ = np.zeros((self.n_clusters, self.n_components))
        for i in range(self.n_clusters):
            self.mu_[i] = trainable_variables[3 + i].detach().numpy()

    def _test_inputs(self, X: NDArray[np.float_]) -> None:
        """
        Just tests to make sure data is numpy array
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")

    def _transform_training_data(self, *args):
        """
        Convert a list of numpy arrays to tensors
        """
        out = []
        for arg in args:
            out.append(torch.tensor(arg))
        return out


def softplus_inverse_np(y):
    """
    Computes the inverse of the softplus activation of x in a
    numerically stable way

    Softplus: y = log(exp(x) + 1)
    Softplus^{-1}: y = np.log(np.exp(x) - 1)

    Parameters
    ----------
    x : np.array
        Original array

    Returns
    -------
    x : np.array
        Transformed array
    """
    min_threshold = 10 ** -15
    max_threshold = 500
    safe_y = np.clip(
        y, min_threshold, max_threshold
    )  # we can safely pass this to the reference inverse_softplus below
    safe_x = np.log(np.exp(safe_y) - 1)

    # if y_i was below (respectively: above) the min (max) threshold, replace with log(y_i)  (y_i)
    x = np.where(y < min_threshold, np.log(y), safe_x)
    x = np.where(y > max_threshold, y, x)
    return x
