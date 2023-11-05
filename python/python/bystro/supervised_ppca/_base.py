"""
This provides the base class of any Gaussian factor model, such as 
(probabilistic) PCA, supervised PCA, and factor analysis, as well as 
supervised and adversarial derivatives. 

Implementing an extension model requires that the following methods be 
implemented
    fit - Learns the model given data (and optionally supervised/adversarial
                       labels
    get_covariance - Given a fitted model returns the covariance matrix

For the remaining shared methods computing likelihoods etc, there are two 
options, compute the covariance matrix then use the default methods from
the covariance module, or use the Sherman-Woodbury matrix identity to 
invert the matrix more efficiently. Currently only the first is implemented
but left the options for future implementation.

Objects
-------
BaseGaussianFactorModel(_BaseSGDModel)
    Base class of all factor analysis models implementing any shared 
    Gaussian methods.

BaseSGDModel(BaseGaussianFactorModel)
    This is the base class of models that use Tensorflow stochastic
    gradient descent for inference. This reduces boilerplate code 
    associated with some of the standard methods etc.

Methods
-------
None
"""
from abc import abstractmethod, ABC

import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray
from typing import Optional

from bystro.covariance._base_covariance import (  # type: ignore
    _get_stable_rank,  # type: ignore
    _conditional_score,  # type: ignore
    _conditional_score_samples,  # type: ignore
    _marginal_score,  # type: ignore
    _marginal_score_samples,  # type: ignore
    _score,  # type: ignore
    _score_samples,  # type: ignore
    _entropy,  # type: ignore
    _entropy_subset,  # type: ignore
    _mutual_information,  # type: ignore
    inv_sherman_woodbury_fa,  # type: ignore
    _get_conditional_parameters_sherman_woodbury,  # type: ignore
    _conditional_score_sherman_woodbury,  # type: ignore
    _conditional_score_samples_sherman_woodbury,  # type: ignore
    _marginal_score_sherman_woodbury,  # type: ignore
    _marginal_score_samples_sherman_woodbury,  # type: ignore
    _score_sherman_woodbury,  # type: ignore
    _score_samples_sherman_woodbury,  # type: ignore
)  # type: ignore
from datetime import datetime as dt
import pytz  # type: ignore
from bystro._template_sgd_np import BaseSGDModel  # type: ignore
import torch  # type: ignore


class BaseGaussianFactorModel(BaseSGDModel, ABC):
    def __init__(self, n_components=2):
        """
        This is the base class of the model. Will never be called directly.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        Sets
        ----
        creationDate : datetime
            The date/time that the object was created
        """
        self.n_components = int(n_components)
        self.W_ = None
        self.creationDate = dt.now(pytz.utc)

    @abstractmethod
    def fit(self, *args):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        other arguments

        Returns
        -------
        self : object
            The model
        """

    @abstractmethod
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

    @abstractmethod
    def get_noise(self):
        """
        Returns the observational noise as a diagnoal matrix

        Parameters
        ----------
        None

        Returns
        -------
        Lambda : np.array-like(p,p)
            The observational noise
        """

    def get_precision(self, sherman_woodbury: bool = False):
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
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            precision = la.inv(covariance)
        else:
            precision = inv_sherman_woodbury_fa(self.get_noise(), self.W_)
        return precision

    def get_stable_rank(self):
        """
        Returns the stable rank defined as
        ||A||_F^2/||A||^2

        Parameters
        ----------
        None

        Returns
        -------
        srank : float
            The stable rank. See Vershynin High dimensional probability for
            discussion, but this is a statistically stable approximation to rank
        """
        covariance = self.get_covariance()
        srank = _get_stable_rank(covariance)
        return srank

    def transform(self, X: NDArray, sherman_woodbury: bool = False):
        """
        This returns the latent variable estimates given X

        Parameters
        ----------
        X : np array-like,(N_samples,p)
            The data to transform.

        Returns
        -------
        S : np.array-like,(N_samples,n_components)
            The factor estimates
        """
        if sherman_woodbury is False:
            prec = self.get_precision()
            coefs = np.dot(self.W_, prec)
        else:
            Lambda = self.get_noise()
            A = la.solve(Lambda, self.W_)
            B = np.dot(A, self.W_.T)
            IpB = np.eye(self.n_components) + B
            end = la.solve(IpB, A)
            coefs = A - np.dot(B, end)

        S = np.dot(X, coefs.T)
        return S

    def transform_subset(
        self,
        X: NDArray,
        observed_feature_idxs: NDArray,
        sherman_woodbury: bool = False,
    ):
        """
        This returns the latent variable estimates given partial observations
        contained in X

        Parameters
        ----------
        X : np array-like,(N_samples,sum(observed_feature_idxs))
            The data to transform.

        observed_feature_idxs: np.array-like,(sum(p),)
            The observation locations

        Returns
        -------
        S : np.array-like,(N_samples,n_components)
            The factor estimates
        """
        if sherman_woodbury is False:
            prec = self.get_precision()
            coefs = np.dot(self.W_, prec)
        else:
            Lambda = self.get_noise()
            coefs, _ = _get_conditional_parameters_sherman_woodbury(
                Lambda, self.W_, observed_feature_idxs
            )

        S = np.dot(X, coefs.T)
        return S

    def conditional_score(
        self,
        X: NDArray,
        observed_feature_idxs: NDArray,
        weights: Optional[NDArray] = None,
        sherman_woodbury: bool = False,
    ):
        """
        Returns the predictive log-likelihood of a subset of data.

        mean(log p(X[idx==1]|X[idx==0],covariance))

        Parameters
        ----------
        X : np.array-like,(N,sum(observed_feature_idxs))
            The data

        observed_feature_idxs: np.array-like,(sum(p),)
            The observation locations

        weights : np.array-like,(N,),default=None
            The optional weights on the samples

        Returns
        -------
        avg_score : float
            Average log likelihood
        """
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            avg_score = _conditional_score(
                covariance, X, observed_feature_idxs, weights=weights
            )
        else:
            avg_score = _conditional_score_sherman_woodbury(
                self.get_noise(),
                self.W_,
                X,
                observed_feature_idxs,
                weights=weights,
            )
        return avg_score

    def conditional_score_samples(
        self,
        X: NDArray,
        observed_feature_idxs: NDArray,
        sherman_woodbury: bool = False,
    ):
        """
        Return the conditional log likelihood of each sample, that is

        log p(X[idx==1]|X[idx==0],covariance)

        Parameters
        ----------
        X : np.array-like,(N,p)
            The data

        observed_feature_idxs: np.array-like,(p,)
            The observation locations

        sherman_woodbury : bool,default=False
            Whether to use the sherman_woodbury matrix identity

        Returns
        -------
        scores : float
            Log likelihood for each sample
        """
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            scores = _conditional_score_samples(
                covariance, X, observed_feature_idxs
            )
        else:
            scores = _conditional_score_samples_sherman_woodbury(
                self.get_noise(), self.W_, X, observed_feature_idxs
            )
        return scores

    def marginal_score(
        self,
        X: NDArray,
        observed_feature_idxs: NDArray,
        weights: Optional[NDArray] = None,
        sherman_woodbury: bool = False,
    ):
        """
        Returns the marginal log-likelihood of a subset of data

        Parameters
        ----------
        X : np.array-like,(N,sum(idxs))
            The data

        observed_feature_idxs: np.array-like,(sum(p),)
            The observation locations

        weights : np.array-like,(N,),default=None
            The optional weights on the samples

        Returns
        -------
        avg_score : float
            Average log likelihood
        """
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            avg_score = _marginal_score(
                covariance, X, observed_feature_idxs, weights=weights
            )
        else:
            avg_score = _marginal_score_sherman_woodbury(
                self.get_noise(),
                self.W_,
                X,
                observed_feature_idxs,
                weights=weights,
            )
        return avg_score

    def marginal_score_samples(
        self,
        X: NDArray,
        observed_feature_idxs: NDArray,
        sherman_woodbury: bool = False,
    ):
        """
        Returns the marginal log-likelihood of a subset of data

        Parameters
        ----------
        X : np.array-like,(N,sum(observed_feature_idxs))
            The data

        observed_feature_idxs: np.array-like,(sum(p),)
            The observation locations

        Returns
        -------
        scores : float
            Average log likelihood
        """
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            scores = _marginal_score_samples(
                covariance, X, observed_feature_idxs
            )
        else:
            scores = _marginal_score_samples_sherman_woodbury(
                self.get_noise(), self.W_, X, observed_feature_idxs
            )
        return scores

    def score(
        self,
        X: NDArray,
        weights: Optional[NDArray] = None,
        sherman_woodbury: bool = False,
    ):
        """
        Returns the average log liklihood of data.

        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data

        weights : np.array-like,(N,),default=None
            The optional weights on the samples

        Returns
        -------
        avg_score : float
            Average log likelihood
        """
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            avg_score = _score(covariance, X, weights=weights)
        else:
            avg_score = _score_sherman_woodbury(
                self.get_noise(), self.W_, X, weights=weights
            )
        return avg_score

    def score_samples(self, X: NDArray, sherman_woodbury: bool = False):
        """
        Return the log likelihood of each sample

        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data

        Returns
        -------
        scores : float
            Log likelihood for each sample
        """
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            scores = _score_samples(covariance, X)
        else:
            scores = _score_samples_sherman_woodbury(
                self.get_noise(), self.W_, X
            )
        return scores

    def get_entropy(self):
        """
        Computes the entropy of a Gaussian distribution parameterized by
        covariance.

        Parameters
        ----------
        None

        Returns
        -------
        entropy : float
            The differential entropy of the distribution
        """
        covariance = self.get_covariance()
        entropy = _entropy(covariance)
        return entropy

    def get_entropy_subset(self, observed_feature_idxs: NDArray):
        """
        Computes the entropy of a subset of the Gaussian distribution
        parameterized by covariance.

        Parameters
        ----------
        observed_feature_idxs: np.array-like,(sum(p),)
            The observation locations

        Returns
        -------
        entropy : float
            The differential entropy of the distribution
        """
        covariance = self.get_covariance()
        entropy = _entropy_subset(covariance, observed_feature_idxs)
        return entropy

    def mutual_information(
        self, observed_feature_idxs1: NDArray, observed_feature_idxs2: NDArray
    ):
        """
        This computes the mutual information bewteen the two sets of
        covariates based on the model.

        Parameters
        ----------
        observed_feature_idxs1 : np.array-like,(p,)
            First group of variables

        observed_feature_idxs2 : np.array-like,(p,)
            Second group of variables

        Returns
        -------
        mutual_information : float
            The mutual information between the two variables
        """
        covariance = self.get_covariance()
        mutual_information = _mutual_information(
            covariance, observed_feature_idxs1, observed_feature_idxs2
        )
        return mutual_information


class BasePCASGDModel(BaseGaussianFactorModel, ABC):
    def __init__(
        self, n_components=2, training_options=None, prior_options=None
    ):
        """
        This is the base class of models that use stochastic
        gradient descent for inference. This reduces boilerplate code
        associated with some of the standard methods etc.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        training_options : dict,default={}
            The options for gradient descent

        prior_options : dict,default={}
            The options for priors on model parameters

        Sets
        ----
        creationDate : datetime
            The date/time that the object was created
        """
        super().__init__(n_components=n_components)

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

    @abstractmethod
    def _create_prior(self):
        """
        Creates a prior on the parameters taking your trainable variable
        dictionary as input

        Parameters
        ----------
        None

        Returns
        -------
        log_prior : function
            The function representing the negative log density of the prior
        """
