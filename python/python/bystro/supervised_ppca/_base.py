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
from datetime import datetime as dt
from typing import Any

import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray

import pytz
import torch
from torch import Tensor

from bystro.covariance._base_covariance import (
    _get_stable_rank,
    _conditional_score,
    _conditional_score_samples,
    _marginal_score,
    _marginal_score_samples,
    _score,
    _score_samples,
    _entropy,
    _entropy_subset,
    _mutual_information,
    inv_sherman_woodbury_fa,
    _get_conditional_parameters_sherman_woodbury,
    _conditional_score_sherman_woodbury,
    _conditional_score_samples_sherman_woodbury,
    _marginal_score_sherman_woodbury,
    _marginal_score_samples_sherman_woodbury,
    _score_sherman_woodbury,
    _score_samples_sherman_woodbury,
)

from bystro._template_sgd_np import BaseSGDModel


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
    def fit(self, *args, **kwargs):
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
    def get_covariance(self) -> NDArray[np.float_]:
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
    def get_noise(self) -> NDArray[np.float_]:
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

    def get_precision(
        self, sherman_woodbury: bool = False
    ) -> NDArray[np.float_]:
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
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return la.inv(covariance)

        return inv_sherman_woodbury_fa(self.get_noise(), self.W_)

    def get_stable_rank(self) -> np.float_:
        """
        Returns the stable rank defined as
        ||A||_F^2/||A||^2

        Parameters
        ----------
        None

        Returns
        -------
        srank : np.float_
            The stable rank. See Vershynin High dimensional probability for
            discussion, but this is a statistically stable approximation to rank
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        return _get_stable_rank(self.get_covariance())

    def transform(self, X: NDArray[np.float_], sherman_woodbury: bool = False):
        """
        This returns the latent variable estimates given X

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

        return np.dot(X, coefs.T)

    def transform_subset(
        self,
        X: NDArray[np.float_],
        observed_feature_idxs: NDArray[np.float_],
        sherman_woodbury: bool = False,
    ) -> NDArray[np.float_]:
        """
        This returns the latent variable estimates given partial observations
        contained in X

        Parameters
        ----------
        X : NDArray,(N_samples,sum(observed_feature_idxs))
            The data to transform.

        observed_feature_idxs: np.array-like,(sum(p),)
            The observation locations

        Returns
        -------
        S : NDArray,(N_samples,n_components)
            The factor estimates
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            prec = self.get_precision()
            coefs = np.dot(self.W_, prec)
        else:
            Lambda = self.get_noise()
            coefs, _ = _get_conditional_parameters_sherman_woodbury(
                Lambda, self.W_, observed_feature_idxs
            )

        return np.dot(X, coefs.T)

    def conditional_score(
        self,
        X: NDArray[np.float_],
        observed_feature_idxs: NDArray[np.float_],
        weights: NDArray[np.float_] | None = None,
        sherman_woodbury: bool = False,
    ) -> np.float_:
        """
        Returns the predictive log-likelihood of a subset of data.

        mean(log p(X[idx==1]|X[idx==0],covariance))

        Parameters
        ----------
        X : NDArray,(N,sum(observed_feature_idxs))
            The data

        observed_feature_idxs: NDArray,(sum(p),)
            The observation locations

        weights : NDArray,(N,),default=None
            The optional weights on the samples

        Returns
        -------
        avg_score : float
            Average log likelihood
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _conditional_score(
                covariance, X, observed_feature_idxs, weights=weights
            )

        return _conditional_score_sherman_woodbury(
            self.get_noise(),
            self.W_,
            X,
            observed_feature_idxs,
            weights=weights,
        )

    def conditional_score_samples(
        self,
        X: NDArray[np.float_],
        observed_feature_idxs: NDArray[np.float_],
        sherman_woodbury: bool = False,
    ) -> np.float_:
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
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _conditional_score_samples(
                covariance, X, observed_feature_idxs
            )

        return _conditional_score_samples_sherman_woodbury(
            self.get_noise(), self.W_, X, observed_feature_idxs
        )

    def marginal_score(
        self,
        X: NDArray[np.float_],
        observed_feature_idxs: NDArray[np.float_],
        weights: NDArray[np.float_] | None = None,
        sherman_woodbury: bool = False,
    ) -> np.float_:
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
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _marginal_score(
                covariance, X, observed_feature_idxs, weights=weights
            )

        return _marginal_score_sherman_woodbury(
            self.get_noise(),
            self.W_,
            X,
            observed_feature_idxs,
            weights=weights,
        )

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
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _marginal_score_samples(covariance, X, observed_feature_idxs)

        return _marginal_score_samples_sherman_woodbury(
            self.get_noise(), self.W_, X, observed_feature_idxs
        )

    def score(
        self,
        X: NDArray[np.float_],
        weights: NDArray[np.float_] | None = None,
        sherman_woodbury: bool = False,
    ) -> np.float_:
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
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            return _score(self.get_covariance(), X, weights=weights)

        return _score_sherman_woodbury(
            self.get_noise(), self.W_, X, weights=weights
        )

    def score_samples(
        self, X: NDArray[np.float_], sherman_woodbury: bool = False
    ) -> np.float_:
        """
        Return the log likelihood of each sample

        Parameters
        ----------
        X : NDArray[np.float_],(N,sum(p))
            The data

        Returns
        -------
        scores : np.float_
            Log likelihood for each sample
        """
        if sherman_woodbury is False:
            return _score_samples(self.get_covariance(), X)

        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        return _score_samples_sherman_woodbury(self.get_noise(), self.W_, X)

    def get_entropy(self) -> np.float_:
        """
        Computes the entropy of a Gaussian distribution parameterized by
        covariance.

        Parameters
        ----------
        None

        Returns
        -------
        entropy : np.float_
            The differential entropy of the distribution
        """
        return _entropy(self.get_covariance())

    def get_entropy_subset(
        self, observed_feature_idxs: NDArray[np.float_]
    ) -> np.float_:
        """
        Computes the entropy of a subset of the Gaussian distribution
        parameterized by covariance.

        Parameters
        ----------
        observed_feature_idxs: NDArray[np.float_],(sum(p),)
            The observation locations

        Returns
        -------
        entropy : np.float_
            The differential entropy of the distribution
        """
        return _entropy_subset(self.get_covariance(), observed_feature_idxs)

    def mutual_information(
        self,
        observed_feature_idxs1: NDArray[np.float_],
        observed_feature_idxs2: NDArray[np.float_],
    ) -> np.float_:
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
        mutual_information : np.float_
            The mutual information between the two variables
        """
        covariance = self.get_covariance()
        return _mutual_information(
            covariance, observed_feature_idxs1, observed_feature_idxs2
        )


class BasePCASGDModel(BaseGaussianFactorModel):
    def __init__(
        self,
        n_components: int = 2,
        training_options: dict[str, Any] | None = None,
        prior_options: dict[str, Any] | None = None,
    ) -> None:
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

    def _fill_training_options(
        self, training_options: dict[str, Any]
    ) -> dict[str, Any]:
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
                "training options were expected but not found..."
            )
        if unexpected_but_present_keys:
            raise ValueError(
                "the following training options were unrecognized but provided..."
            )

        return tops

    def _initialize_save_losses(self) -> None:
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

    def _save_losses(
        self,
        i: int,
        log_likelihood: Tensor,
        log_prior: Tensor | NDArray[np.float_],
        log_posterior: Tensor,
    ) -> None:
        """
        Saves the values of the losses at each iteration

        Parameters
        -----------
        i : int
            Current training iteration

        losses_likelihood : Tensor
            The log likelihood

        losses_prior : Tensor | NDArray[np.float_]
            The log prior

        losses_posterior : Tensor
            The log posterior
        """
        self.losses_likelihood[i] = log_likelihood.detach().numpy()
        if isinstance(log_prior, np.ndarray):
            self.losses_prior[i] = log_prior
        else:
            self.losses_prior[i] = log_prior.detach().numpy()
        self.losses_posterior[i] = log_posterior.detach().numpy()

    def _transform_training_data(self, *args: NDArray) -> list[Tensor]:
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
