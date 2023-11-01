"""

Objects
-------


Methods
-------


"""
import numpy as np
from bystro.covariance._base_covariance import (
    _conditional_score,
)
from bystro._template_sgd_np import BaseSGDModel  # type: ignore
import torch
from torch import nn


class GaussianMixturePPCA(BaseSGDModel):
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

        method : string {'Nadam'}, default='Nadam'
            The learning algorithm

        batch_size : int, default=None
            The number of observations to use at each iteration. If none
            corresponds to batch learning

        gpu_memory : int, default=1024
            The amount of memory you wish to use during training
        """
        default_options = {
            "n_iterations": 3000,
            "learning_rate": 1e-2,
            "gpu_memory": 1024,
            "method": "Nadam",
            "batch_size": 100,
            "momentum": 0.9,
        }
        tops = {**default_options, **training_options}

        default_keys = set(default_options.keys())
        final_keys = set(tops.keys())

        expected_but_missing_keys = default_keys - final_keys
        unexpected_but_present_keys = final_keys - default_keys
        if expected_but_missing_keys:
            raise ValueError(
                "the following training options were expected but not found..."
            )
        if unexpected_but_present_keys:
            raise ValueError(
                "the following training options were unrecognized but provided..."
            )
        return tops

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
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        rng = np.random.default_rng(int(seed))
        K = self.n_clusters

        W_, sigmal_, pi_, mu_list = self._initialize_variables(X)

        X = self._transform_training_data(X)

        trainable_variables = [W_, sigmal_, pi_] + mu_list

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        softplus = nn.Softplus()
        mse = nn.MSELoss()
        smax = nn.Softmax()

        myrange = trange if progress_bar else range
        eye = torch.tensor(np.eye(p).astype(np.float32))

        for i in myrange(training_options["n_iterations"]):
            idx = rng.choice(
                X.shape[0], size=training_options["batch_size"], replace=False
            )

            X_batch = X[idx]

            sigma2 = softplus(sigmal_)

            X_each = [X_batch - torch.matmul(W_, mu_list[k]) for k in range(K)]

            Sigma = torch.matmul(torch.transpose(W_, 0, 1), W_) + sigma2 * eye

            m = MultivariateNormal(torch.zeros(p), Sigma)

            pi = smax(pi_logits)
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

    def _transform_training_data(self, *args):
        """
        Convert a list of numpy arrays to tensors
        """
        out = []
        for arg in args:
            out.append(torch.tensor(arg))
        return out
