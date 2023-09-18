"""
This provides the base class for covariance estimation. While there are 
many methods for estimating covariance matrices, once such a matrix has 
been estimated most of the procedures for evaulating fit, comparing 
quantities etc are shared between models. This implementation allows for 
us to implement these methods a single time and for most code being 
centered on the fit method and a small number of auxiliary methods. This 
also will provide the basis for factor analysis and probablistic PCA. As 
a practical design choice, these methods are implemented as separate 
functions, which are simply converted into object methods by replacing 
covariance with self.covariance. Documentation is provided in the 
functional method and omitted from the object methods.

Objects
-------
BaseCovariance

    get_precision()
        Gets the precision matrix defined as the inverse of the covariance 

    get_stable_rank()
        Returns the stable rank defined as 
        ||A||_F^2/||A||^2

    predict(Xobs,idxs)
        Predicts missing data using observed data.

    --------------------
    conditional_score(X,idxs,weights=None)
        mean(log p(X[idx==1]|X[idx==0],covariance))

    conditional_score_samples(X,idxs)
        log p(X[idx==1]|X[idx==0],covariance)
    
    marginal_score(X,idxs,weights=None)
        mean(log p(X[idx==1]|covariance))

    marginal_score_samples(X,idxs)
        log p(X[idx==1]|covariance)

    score(X,weights=None):
        mean log p(X)

    score_samples(X)
        log p(X)

    --------------------
    entropy()
        Computes the entropy of a Gaussian distribution parameterized by 
        covariance.
    
    entropy_subset(idxs)
        Computes the entropy of a subset of the covariates

    mutual_information(covariance,idxs1,idxs2):
        This computes the mutual information bewteen the two sets of
        covariates based on the model.

Methods
-------
_get_precision(covariance)

_get_stable_rank(covariance)

_predict(covariance,Xobs,idxs)

_get_conditional_parameters(covariance,idxs)
    Computes the distribution parameters p(X_miss|X_obs)

----------------------------

_conditional_score_samples(covariance,X,idxs,weights=None)

_conditional_score(covariance,X,idxs)

_marginal_score(covariance,X,idxs,weights=None)

_marginal_score_samples(covariance,X,idxs)

_score(covariance,X,weights=None)

_score_samples(covariance,X)

----------------------------

_entropy(covariance)

_entropy_subset(covariance,idxs)

_mutual_information(covariance,idxs1,idxs2):

"""
import numpy as np
from numpy import linalg as la
from datetime import datetime as dt
import pytz


class BaseCovariance(object):
    """
    This object basically contains all the methods asides for fitting
    a covariance matrix.
    """

    def __init__(self):
        self.creationDate = dt.now(pytz.timezone("US/Pacific"))

    #################
    # Miscellaneous #
    #################
    def get_precision(self):
        return _get_precision(self.covariance)

    def get_stable_rank(self):
        return _get_stable_rank(self.covariance)

    def predict(self, Xobs, idxs):
        return _predict(self.covariance, Xobs, idxs)

    #####################################
    # Gaussian Likelihood-based methods #
    #####################################
    def conditional_score(self, X, idxs, weights=None):
        return _conditional_score(self.covariance, X, idxs, weights=weights)

    def conditional_score_samples(self, X, idxs):
        return _conditional_score_samples(self.covariance, X, idxs)

    def marginal_score(self, X, idxs, weights=None):
        return _marginal_score(self.covariance, X, idxs, weights=weights)

    def marginal_score_samples(self, X, idxs):
        return _marginal_score_samples(self.covariance, X, idxs)

    def score(self, X, weights=None):
        return _score(self.covariance, X, weights=weights)

    def score_samples(self, X):
        return _score_samples(self.covariance, X)

    #################################
    # Information-theoretic methods #
    #################################
    def entropy(self):
        return _entropy(self.covariance)

    def entropy_subset(self, idxs):
        return _entropy_subset(self.covariance, idxs)

    def mutual_information(self, idxs1, idxs2):
        return _mutual_information(self.covariance, idxs1, idxs2)


###########################################
###########################################
###########################################
###                                     ###
###                                     ###
###   Methods for covariance matrices   ###
###                                     ###
###                                     ###
###########################################
###########################################
###########################################


def _get_precision(covariance):
    """
    Gets the precision matrix defined as the inverse of the covariance

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    Returns
    -------
    precision : np.array-like(sum(p),sum(p))
        The inverse of the covariance matrix
    """
    precision = la.inv(covariance)
    return precision


def _get_stable_rank(covariance):
    """
    Returns the stable rank defined as
    ||A||_F^2/||A||^2

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    Returns
    -------
    srank : float
        The stable rank. See Vershynin High dimensional probability for 
        discussion, but this is a statistically stable approximation to rank
    """
    singular_values = la.svd(covariance, compute_uv=False)
    srank = np.sum(singular_values) / singular_values[0]
    return srank


def _predict(covariance, Xobs, idxs):
    """
    Predicts missing data using observed data. This uses the conditional
    Gaussian formula (see wikipedia multivariate gaussian). Solve allows
    us to avoid an explicit matrix inversion.

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    Xobs : np array-like,(N_samples,\\sum idxs)
        The observed data

    idxs: np.array-like,(sum(p),)
        The observation locations

    Returns
    -------
    preds : np.array-like,(N_samples,p-\\sum idxs)
        The predicted values
    """
    covariance_sub = covariance[idxs == 1]
    covariance_22 = covariance_sub[:, idxs == 1]
    covariance_21 = covariance_sub[:, idxs == 0]

    beta_bar = la.solve(covariance_22, covariance_21)

    preds = np.dot(Xobs[:, idxs == 1], beta_bar)
    return preds


def _conditional_score(covariance, X, idxs, weights=None):
    """
    Returns the predictive log-likelihood of a subset of data.

    mean(log p(X[idx==1]|X[idx==0],covariance))

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    X : np.array-like,(N,sum(idxs))
        The centered data

    idxs: np.array-like,(sum(p),)
        The observation locations

    weights : np.array-like,(N,),default=None
        The optional weights on the samples. Don't have negative values. 
        Average value forced to 1.

    Returns
    -------
    avg_score : float
        Average log likelihood
    """
    weights = (
        np.ones(X.shape[0]) if weights is None else weights / np.mean(weights)
    )
    avg_score = np.mean(
        weights * _conditional_score_samples(covariance, X, idxs)
    )
    return avg_score


def _conditional_score_samples(covariance, X, idxs):
    """
    Return the conditional log likelihood of each sample, that is

    log p(X[idx==1]|X[idx==0],covariance) = N(Sigma_10Sigma_00^{-1}x_0,
                                    Sigma_11-Sigma_10Sigma_00^{-1}Sigma_01)

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    X : np.array-like,(N,p)
        The centered data

    idxs: np.array-like,(p,)
        The observation locations

    Returns
    -------
    scores : float
        Log likelihood for each sample
    """
    covariance_sub = covariance[idxs == 1]
    covariance_22 = covariance_sub[:, idxs == 1]
    covariance_21 = covariance_sub[:, idxs == 0]

    covariance_nonsub = covariance[idxs == 0]
    covariance_11 = covariance_nonsub[:, idxs == 0]

    beta_bar = la.solve(covariance_22, covariance_21)
    Second_part = np.dot(covariance_21.T, beta_bar)

    covariance_bar = covariance_11 - Second_part

    mu_ = np.dot(X[:, idxs == 1], beta_bar)
    scores = _score_samples(covariance_bar, X[:, idxs == 0] - mu_)

    return scores


def _get_conditional_parameters(covariance, idxs):
    """
    Computes the distribution parameters p(X_miss|X_obs)

    Parameters
    ----------
    covariance : np.array-like,(p,p)
        The covariance matrix

    idxs: np.array-like,(p,)
        The observed covariates

    Returns
    -------
    beta_bar : array-like
        The predictive covariates

    covariance_bar : array-like
        Conditional covariance
    """

    covariance_sub = covariance[idxs == 1]
    covariance_22 = covariance_sub[:, idxs == 1]
    covariance_21 = covariance_sub[:, idxs == 0]
    covariance_nonsub = covariance[idxs == 0]
    covariance_11 = covariance_nonsub[:, idxs == 0]
    beta_bar = la.solve(covariance_22, covariance_21)
    Second_part = np.dot(covariance_21.T, beta_bar)

    covariance_bar = covariance_11 - Second_part
    return beta_bar.T, covariance_bar


def _marginal_score(covariance, X, idxs, weights=None):
    """
    Returns the marginal log-likelihood of a subset of data

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    X : np.array-like,(N,sum(idxs))
        The centered data

    idxs: np.array-like,(sum(p),)
        The observation locations

    weights : np.array-like,(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : float
        Average log likelihood
    """
    if weights is None:
        weights = np.ones(X.shape[0])
    avg_score = np.mean(weights * _marginal_score_samples(covariance, X, idxs))
    return avg_score


def _marginal_score_samples(covariance, X, idxs):
    """
    Returns the marginal log-likelihood of a subset of data

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    X : np.array-like,(N,sum(idxs))
        The centered data

    idxs: np.array-like,(sum(p),)
        The observation locations

    Returns
    -------
    scores : float
        Average log likelihood
    """
    cov1 = covariance[idxs == 1]
    cov_sub = cov1[:, idxs == 1]
    scores = _score_samples(cov_sub, X)
    return scores


def _score(covariance, X, weights=None):
    """
    Returns the average log liklihood of data.

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    X : np.array-like,(N,sum(p))
        The centered data

    weights : np.array-like,(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : float
        Average log likelihood
    """
    if weights is None:
        weights = np.ones(X.shape[0])
    avg_score = np.mean(weights * _score_samples(covariance, X))
    return avg_score


def _score_samples(covariance, X):
    """
    Return the log likelihood of each sample

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    X : np.array-like,(N,sum(p))
        The centered data

    Returns
    -------
    scores : float
        Log likelihood for each sample
    """
    p = covariance.shape[0]
    term1 = -p / 2 * np.log(2 * np.pi)

    _, logdet = la.slogdet(covariance)
    term2 = -0.5 * logdet

    quad_init = la.solve(covariance, np.transpose(X))
    difference = X * np.transpose(quad_init)
    term3 = np.sum(difference, axis=1)

    scores = term1 + term2 - 0.5 * term3
    return scores


def _entropy(covariance):
    """
    Computes the entropy of a Gaussian distribution parameterized by
    covariance.

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    Returns
    -------
    entropy : float
        The differential entropy of the distribution
    """
    cov_new = 2 * np.pi * np.e * covariance
    _, logdet = la.slogdet(cov_new)
    entropy = 0.5 * logdet
    return entropy


def _entropy_subset(covariance, idxs):
    """
    Computes the entropy of a subset of the Gaussian distribution
    parameterized by covariance.

    Parameters
    ----------
    covariance : np.array-like(p,p)
        The covariance matrix

    idxs: np.array-like,(sum(p),)
        The observation locations

    Returns
    -------
    entropy : float
        The differential entropy of the distribution
    """
    cov1 = covariance[idxs == 1]
    cov_sub = cov1[:, idxs == 1]
    entropy = _entropy(cov_sub)
    return entropy


def _mutual_information(covariance, idxs1, idxs2):
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
    idxs = idxs1 + idxs2
    cov_sub = covariance[np.ix_(idxs == 1, idxs == 1)]
    idxs1_sub = idxs1[idxs == 1]
    Hy = _entropy(cov_sub[np.ix_(idxs1_sub == 1, idxs1_sub == 1)])

    _, covariance_conditional = _get_conditional_parameters(
        cov_sub, 1 - idxs1_sub
    )
    H_y_given_x = _entropy(covariance_conditional)
    mutual_information = Hy - H_y_given_x
    return mutual_information
