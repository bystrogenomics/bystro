"""
This implements Gaussian factor analysis models with generative (standard)
inference. There are three different models. Gaussian factor analysis 
parameterizes the covariance matrix as 

    Sigma = WW^T + Lambda

where Lambda is a diagonal matrix. The parameters of the diagonal determine
the model. Probabilistic principal component analysis sets Lambda =
sigma**2*I_p. Supervised principal component analysis parameterizes it as 
Lambda = diag([sigma_x**2*1_p,sigma_y**21_q]), allowing for variances to 
differ between the predictive and dependent variables. Finally, factor 
analysis allows each diagonal component to be distinct. Models (1) and (3)
are described in Bishop 2006, while supervised Probabilistic PCA is 
described in several papers, including Yu 2006.

Objects
-------
PPCApt(BaseGDModel)
    Principal component analysis but with tensorflow implementation.

SPCAtf(BaseGDModel)
    Supervised probabilistic component analysis

FactorAnalysistf(BaseGDModel)
    Factor analysis implemented in tensorflow. See Bishop 2006

Methods
-------
None
"""
import numpy as np

from sklearn import decomposition as dp
from tqdm import trange
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from ._misc_np import softplus_inverse_np
from ._base import BaseGDModel
from copy import deepcopy


class PPCApt(BaseGDModel):
    def __init__(self, n_components=2, prior_options={}, training_options={}):
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

    def __repr__(self):
        out_str = "PPCApt object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X, progress_bar=True):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : None
            Used for model consistency

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        Returns
        -------
        self : object
            The model
        """
        td = self.training_options
        N, p = X.shape
        self.p = p
        rng = np.random.default_rng(2021)

        W_, sigmal_ = self._initialize_variables(X)
        X = torch.tensor(X)

        trainable_variables = [W_, sigmal_]

        self._initialize_saved_losses()

        optimizer = torch.optim.SGD(
            trainable_variables, lr=td["learning_rate"], momentum=0.9
        )
        eye = torch.tensor(np.eye(p).astype(np.float32))
        softplus = nn.Softplus()

        _prior = self._create_prior()

        myrange = trange if progress_bar else range

        for i in myrange(td["n_iterations"]):
            idx = rng.choice(X.shape[0], size=td["batch_size"], replace=False)
            X_batch = X[idx]

            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            m = MultivariateNormal(torch.zeros(p), Sigma)

            like_tot = torch.mean(m.log_prob(X_batch))
            like_prior = _prior(W_, sigma)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, like_tot, like_prior, posterior)

        self._save_variables(trainable_variables)

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

        def log_prior(W, sigma):
            return 0.0 * torch.mean(W) + 0.0 * sigma

        return log_prior

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

        losses_likelihood : tf.Float
            The log likelihood

        losses_prior : tf.Float
            The log prior

        losses_posterior : tf.Float
            The log posterior
        """
        self.losses_likelihood[i] = log_likelihood.detach().numpy()
        if torch.is_tensor(log_prior):
            self.losses_prior[i] = log_prior.detach().numpy()
        else:
            self.losses_prior[i] = log_prior
        self.losses_posterior[i] = log_posterior.detach().numpy()

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

        sigma2_ : float
            The isotropic variance
        """
        self.W_ = trainable_variables[0].detach().numpy()
        self.sigma2_ = nn.Softplus()(trainable_variables[1]).detach().numpy()

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        default = {}
        new_dict = deepcopy(default)
        new_dict.update(prior_options)
        return new_dict


class SPCApt(BaseGDModel):
    def __init__(self, n_components=2, prior_options={}, training_options={}):
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

    def __repr__(self):
        out_str = "SPCApt object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X, groups=None):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : None
            Used for model consistency

        groups : np.array-like,(n_covariates,)
            Divide the covariates into groups with different isotropic noise

        Returns
        -------
        self : object
            The model
        """

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

        sigmal_ : list[tf.Variable]
            A list of the isotropic noises for each group
        """

    def _save_variables(self, W_, D_):
        """
        Saves the learned variables

        Parameters
        ----------
        W_ : tf.Tensor,(n_components,p)
            The loadings

        D_ : tf.Tensor,(p,p)
            Diagonal matrix of isotropic noises

        Sets
        ----
        W_ : np.array-like,(n_components,p)
            The loadings

        sigmas_ : np.array-like,(n_components,p)
            The diagonal variances
        """


class FactorAnalysispt(BaseGDModel):
    def __init__(self, n_components=2, prior_options={}, training_options={}):
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
        super(FactorAnalysispt, self).__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )

    def __repr__(self):
        out_str = "FactorAnalysispt object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X, groups=None):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : None
            Used for model consistency

        Returns
        -------
        self : object
            The model
        """

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

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the negative log density of the prior
        """

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """

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
