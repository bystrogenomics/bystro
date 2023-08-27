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

BaseGDModel(BaseGaussianFactorModel)
    This is the base class of models that use Tensorflow stochastic
    gradient descent for inference. This reduces boilerplate code 
    associated with some of the standard methods etc.

Methods
-------
None
"""
from abc import abstractmethod
import numpy as np
from ..covariance._base_covariance import (
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
)
from numpy import linalg as la
from datetime import datetime as dt
from .._template_sgd_np import _BaseSGDModel
from copy import deepcopy


class BaseGaussianFactorModel(_BaseSGDModel):
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
        self.creationDate = dt.now(dt.timezone.utc)

    @abstractmethod
    def fit(self, X, y=None, groups=None):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : np.array,default=None
            Optional labels

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
    def _save_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of tensorflow variables saved
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
            The stable rank
        """
        covariance = self.get_covariance()
        srank = _get_stable_rank(covariance)
        return srank

    def transform(self, X):
        """
        This returns the latent variable estimates given X

        Parameters
        ----------
        X : np array-like,(N_samples,\sum idxs)
            The data to transform

        Returns
        -------
        S : np.array-like,(N_samples,n_components)
            The factor estimates
        """
        prec = self.get_precision()
        coefs = np.dot(self.W_, prec)
        S = np.dot(X, coefs.T)
        return S

    #####################################
    # Gaussian Likelihood-based methods #
    #####################################
    def conditional_score(self, X, idxs, weights=None, sherman_woodbury=False):
        """
        Returns the predictive log-likelihood of a subset of data.

        mean(log p(X[idx==1]|X[idx==0],covariance))

        Parameters
        ----------
        X : np.array-like,(N,sum(idxs))
            The data

        idxs: np.array-like,(sum(p),)
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
            avg_score = _conditional_score(covariance, X, idxs, weights=weights)
        else:
            raise NotImplementedError("Sherman woodbury not imlemented yet")
        return avg_score

    def conditional_score_samples(self, X, idxs, sherman_woodbury=False):
        """
        Return the conditional log likelihood of each sample, that is

        log p(X[idx==1]|X[idx==0],covariance)

        Parameters
        ----------
        X : np.array-like,(N,p)
            The data

        idxs: np.array-like,(p,)
            The observation locations

        Returns
        -------
        scores : float
            Log likelihood for each sample
        """
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            scores = _conditional_score_samples(covariance, X, idxs)
        else:
            raise NotImplementedError("Sherman woodbury not imlemented yet")
        return scores

    def marginal_score(self, X, idxs, weights=None, sherman_woodbury=False):
        """
        Returns the marginal log-likelihood of a subset of data

        Parameters
        ----------
        X : np.array-like,(N,sum(idxs))
            The data

        idxs: np.array-like,(sum(p),)
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
            avg_score = _marginal_score(covariance, X, idxs, weights=weights)
        else:
            raise NotImplementedError("Sherman woodbury not imlemented yet")
        return avg_score

    def marginal_score_samples(self, X, idxs, sherman_woodbury=False):
        """
        Returns the marginal log-likelihood of a subset of data

        Parameters
        ----------
        X : np.array-like,(N,sum(idxs))
            The data

        idxs: np.array-like,(sum(p),)
            The observation locations

        Returns
        -------
        scores : float
            Average log likelihood
        """
        if sherman_woodbury is False:
            self.get_covariance()
            scores = _marginal_score_samples(X, idxs)
        else:
            raise NotImplementedError("Sherman woodbury not imlemented yet")
        return scores

    def score(self, X, weights=None, sherman_woodbury=False):
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
            raise NotImplementedError("Sherman woodbury not imlemented yet")
        return avg_score

    def score_samples(self, X, sherman_woodbury=False):
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
            raise NotImplementedError("Sherman woodbury not imlemented yet")
        return scores

    def entropy(self):
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

    def entropy_subset(self, idxs):
        """
        Computes the entropy of a subset of the Gaussian distribution
        parameterized by covariance.

        Parameters
        ----------
        idxs: np.array-like,(sum(p),)
            The observation locations

        Returns
        -------
        entropy : float
            The differential entropy of the distribution
        """
        covariance = self.get_covariance()
        entropy = _entropy_subset(covariance, idxs)
        return entropy

    def mutual_information(self, idxs1, idxs2):
        """
        This computes the mutual information bewteen the two sets of
        covariates based on the model.

        Parameters
        ----------
        idxs1 : np.array-like,(p,)
            First group of variables

        idxs2 : np.array-like,(p,)
            Second group of variables

        Returns
        -------
        mutual_information : float
            The mutual information between the two variables
        """
        covariance = self.get_covariance()
        mutual_information = _mutual_information(covariance, idxs1, idxs2)
        return mutual_information


class BaseGDModel(BaseGaussianFactorModel):
    def __init__(self, n_components=2, training_options={}, prior_options={}):
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
        }
        tops = deepcopy(default_options)
        tops.update(training_options)
        return tops

    def _initialize_saved_losses(self):
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
        self.losses_likelihood = np.zeros(n_iterations)
        self.losses_prior = np.zeros(n_iterations)
        self.losses_posterior = np.zeros(n_iterations)

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
        self.losses_likelihood[i] = log_likelihood.numpy()
        if isinstance(log_prior, np.ndarray):
            self.losses_prior[i] = log_prior
        else:
            self.losses_prior[i] = log_prior.numpy()
        self.losses_posterior[i] = log_posterior.numpy()

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


"""
MIT License

Copyright (c) 2022 Austin Talbot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
