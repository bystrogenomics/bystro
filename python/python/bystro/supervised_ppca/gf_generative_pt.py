"""
This implements Gaussian factor analysis models with generative (standard)
inference. There are three different models. Gaussian factor analysis 
parameterizes the covariance matrix as 

    Sigma = WW^T + Lambda

where Lambda is a diagonal matrix (see Bishop 2006 Chapter 12 for notation
and model definition).  Probabilistic principal component analysis 
sets Lambda = sigma**2*I_p. Supervised principal component analysis 
parameterizes it as 
Lambda = diag([sigma_x**2*1_p,sigma_y**2 1_q]), allowing for variances to 
differ between the predictive and dependent variables. Finally, factor 
analysis allows each diagonal component to be distinct. Models (1) and (3)
are described in Bishop 2006, while supervised Probabilistic PCA is 
described in several papers, including Yu 2006.

Objects
-------
PPCApt(BaseSGDModel)
    Principal component analysis but with Pytorch implementation.

SPCApt(BaseSGDModel)
    Supervised probabilistic component analysis

FactorAnalysispt(BaseSGDModel)
    Factor analysis implemented in pytorch. See Bishop 2006

Methods
-------
None
"""
import numpy as np

from sklearn.decomposition import PCA  # type: ignore
from tqdm import trange  # type: ignore
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma

from bystro.supervised_ppca._misc_np import softplus_inverse_np
from bystro.supervised_ppca._base import BasePCASGDModel


class PPCA(BasePCASGDModel):
    def __init__(
        self, n_components=2, prior_options=None, training_options=None
    ):
        """
        This implements probabilistic PCA with stochastic gradient descent.
        There are two benefits over the standard baseline method (1) it
        allows for priors to be placed on the parameters. (2) is minor but
        it theoretically allows for larger datsets that can't be loaded into
        memory. More importantly, it makes a fairer baseline comparison
        for my other models because it allows for stochastic noise.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        training_options : dict,default={}
            The options for gradient descent

        prior_options : dict,default={}
            The options for priors on model parameters
        """
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()

    def __repr__(self):
        return f"PPCApt(n_components={self.n_components})"

    def fit(self, X, progress_bar=True, seed=2021):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

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
        self._test_inputs(X)
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        rng = np.random.default_rng(int(seed))

        W_, sigmal_ = self._initialize_variables(X)

        X = self._transform_training_data(X)[0]

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        eye = torch.tensor(np.eye(p).astype(np.float32))
        softplus = nn.Softplus()

        _prior = self._create_prior()

        myrange = trange if progress_bar else range

        for i in myrange(training_options["n_iterations"]):
            idx = rng.choice(
                X.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X[idx]

            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            m = MultivariateNormal(torch.zeros(p), Sigma)

            like_tot = torch.mean(m.log_prob(X_batch))
            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, like_tot, like_prior, posterior)

        self._store_instance_variables(trainable_variables)

        return self

    def get_covariance(self):
        """
        Gets the covariance matrix

        Sigma = W^TW + sigma2*I

        Parameters
        ----------
        None

        Returns
        -------
        covariance : np.array-like(p,p)
            The covariance matrix
        """
        covariance = np.dot(self.W_.T, self.W_) + self.sigma2_ * np.eye(self.p)
        return covariance

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the log density of the prior
        """
        prior_options = self.prior_options

        def log_prior(trainable_variables):
            W_ = trainable_variables[0]
            sigma_ = trainable_variables[1]
            part1 = (
                -1 * prior_options["weight_W"] * torch.mean(torch.square(W_))
            )
            part2 = Gamma(
                prior_options["alpha"], prior_options["beta"]
            ).log_prob(sigma_)
            out = torch.mean(part1 + part2)
            return out

        return log_prior

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
        """
        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = torch.tensor(sinv, requires_grad=True)
        return W_, sigmal_

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
        if torch.is_tensor(log_prior):
            self.losses_prior[i] = log_prior.detach().numpy()
        else:
            self.losses_prior[i] = log_prior
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

    def _test_inputs(self, X):
        """
        Just tests to make sure data is numpy array
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        default_dict = {"weight_W": 0.01, "alpha": 3.0, "beta": 3.0}
        new_dict = {**default_dict, **prior_options}
        return new_dict


class SPCA(BasePCASGDModel):
    def __init__(
        self, n_components=2, prior_options=None, training_options=None
    ):
        """
        This implements supervised probabilistic component analysis. Unlike
        PPCA there are no analytic solutions for this model. While the
        initial paper used expectation maximization as an inference method,
        EM is actually pretty bad so this is a way better way to go.

        SPPCA replaces isotropic noise with noise for groups of variables.
        The paper Yu et al (2006) only has two groups of covariates, but
        my implementation is more general in that it allows for multiple
        groups rather than two.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        training_options : dict,default={}
            The options for gradient descent

        prior_options : dict,default={}
            The options for priors on model parameters
        """
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()

    def __repr__(self):
        return f"SPCApt(n_components={self.n_components})"

    def fit(self, X, groups, progress_bar=True, seed=2021):
        """
        Fits a model given covariates X 

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        groups : np.array-like,(n_covariates,)
            Divide the covariates into groups with different isotropic noise

        Returns
        -------
        self : object
            The model
        """
        self._test_inputs(X, groups)
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        self.n_groups = len(np.unique(groups))
        rng = np.random.default_rng(int(seed))

        W_, sigmals_ = self._initialize_variables(X)

        X = self._transform_training_data(X)[0]

        self.groups = groups
        list_constants = []
        for i in range(self.n_groups):
            torch_array = torch.zeros(self.p)
            torch_array[groups == i] = 1
            list_constants.append(torch_array)

        trainable_variables = [W_] + sigmals_

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        softplus = nn.Softplus()

        _prior = self._create_prior()

        myrange = trange if progress_bar else range

        for i in myrange(training_options["n_iterations"]):
            idx = rng.choice(
                X.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X[idx]

            list_covs = [
                softplus(sigmals_[k]) * list_constants[k]
                for k in range(self.n_groups)
            ]

            sigma = torch.sum(list_covs, dim=0,)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + torch.diag(sigma)

            m = MultivariateNormal(torch.zeros(p), Sigma)

            like_tot = torch.mean(m.log_prob(X_batch))
            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, like_tot, like_prior, posterior)

        self._store_instance_variables(trainable_variables)

        return self

    def get_covariance(self):
        """
        Gets the covariance matrix

        Sigma = W^TW + sigma2*I

        Parameters
        ----------
        None

        Returns
        -------
        covariance : np.array-like(p,p)
            The covariance matrix
        """
        covariance = np.dot(self.W_.T, self.W_) + np.diag(self.sigmas_)
        return covariance

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the negative log density of the prior
        """
        prior_options = self.prior_options

        def log_prior(trainable_variables):
            list_gamma_log_probs = []
            for k in self.n_groups:
                list_gamma_log_probs.append(
                    Gamma(
                        prior_options["alpha"], prior_options["beta"]
                    ).log_prob(trainable_variables[k + 1])
                )

            gamma_log_probs = torch.stack(list_gamma_log_probs)
            return torch.mean(gamma_log_probs)

        return log_prior

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        default_dict = {"alpha": 3.0, "beta": 3.0}
        new_dict = {**default_dict, **prior_options}
        return new_dict

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
        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = [
            torch.tensor(sinv, requires_grad=True) for i in range(self.n_groups)
        ]
        return W_, sigmal_

    def _store_instance_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of variables learned by pytorch

        Sets
        ----
        W_ : np.array-like,(n_components,p)
            The loadings

        sigmas_ : np.array-like,(p,)
            The diagonal variances
        """
        self.W_ = trainable_variables[0].detach().numpy()
        self.sigmas_ = np.zeros(self.p)
        for k in range(self.n_groups):
            sigma2 = nn.Softplus()(trainable_variables[k + 1])
            self.sigmas_[self.groups == k] = sigma2

    def _test_inputs(self, X, groups):
        """
        Just tests to make sure data is numpy array
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X is numpy array")
        if not isinstance(groups, np.ndarray):
            raise ValueError("groups is numpy array")
        if X.shape[1] != len(groups):
            raise ValueError("Dimensions do not match")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")


class FactorAnalysis(BasePCASGDModel):
    def __init__(
        self, n_components=2, prior_options=None, training_options=None
    ):
        """
        This implements factor analysis which allows for each covariate to
        have it's own isotropic noise. No analytic solution that I know of
        but fortunately with SGD it doesn't matter.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        training_options : dict,default={}
            The options for gradient descent

        prior_options : dict,default={}
            The options for priors on model parameters
        """
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()

    def __repr__(self):
        return f"FactorAnalysispt(n_components={self.n_components})"

    def fit(self, X, progress_bar=True, seed=2021):
        """
        Fits a model given covariates X 

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        Returns
        -------
        self : object
            The model
        """
        self._test_inputs(X)
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        rng = np.random.default_rng(int(seed))

        W_, sigmal_ = self._initialize_variables(X)

        X = self._transform_training_data(X)[0]

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        softplus = nn.Softplus()

        _prior = self._create_prior()

        myrange = trange if progress_bar else range

        for i in myrange(training_options["n_iterations"]):
            idx = rng.choice(
                X.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X[idx]

            sigmas = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            D = torch.diag(sigmas)
            Sigma = WWT + D

            m = MultivariateNormal(torch.zeros(p), Sigma)

            like_tot = torch.mean(m.log_prob(X_batch))
            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, like_tot, like_prior, posterior)

        self._store_instance_variables(trainable_variables)

        return self

    def get_covariance(self):
        """
        Gets the covariance matrix

        Sigma = W^TW + sigma2*I

        Parameters
        ----------
        None

        Returns
        -------
        covariance : np.array-like(p,p)
            The covariance matrix
        """
        covariance = np.dot(self.W_.T, self.W_) + np.diag(self.sigmas_)
        return covariance

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the negative log density of the prior
        """
        prior_options = self.prior_options

        def log_prior(trainable_variables):
            sigma_ = nn.Softmax()(trainable_variables[1])
            return torch.mean(
                Gamma(prior_options["alpha"], prior_options["beta"]).log_prob(
                    sigma_
                )
            )

        return log_prior

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        default_dict = {"alpha": 3.0, "beta": 3.0}
        new_dict = {**default_dict, **prior_options}
        return new_dict

    def _initialize_variables(self, X):
        """
        Initializes the variables of the model by fitting PCA model in 
        sklearn and using those loadings

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        Returns
        -------
        W_ : torch.tensor,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : torch.tensor,(p,)
            The noise of each covariate, unrectified
        """
        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = torch.tensor(
            sinv[0] * np.ones(self.p).astype(np.float32), requires_grad=True
        )
        return W_, sigmal_

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

        sigmas_ : np.array-like,(n_components,p)
            The diagonal variances
        """
        self.W_ = trainable_variables[0].detach().numpy()
        self.sigmas_ = nn.Softplus()(trainable_variables[1]).detach().numpy()

    def _test_inputs(self, X):
        """
        Just tests to make sure data is numpy array
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
