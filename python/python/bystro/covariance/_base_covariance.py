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
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numpy import linalg as la
from datetime import datetime as dt
import pytz


class BaseCovariance:
    """
    This object basically contains all the methods asides for fitting
    a covariance matrix.
    """

    def __init__(self):
        self.creationDate = dt.now(pytz.timezone("US/Pacific"))
        self.covariance: NDArray | None = None

    #################
    # Miscellaneous #
    #################
    def get_precision(self) -> NDArray[np.float_]:
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _get_precision(self.covariance)

    def get_stable_rank(self):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _get_stable_rank(self.covariance)

    def predict(self, Xobs: NDArray, idxs: NDArray):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _predict(self.covariance, Xobs, idxs)

    #####################################
    # Gaussian Likelihood-based methods #
    #####################################
    def conditional_score(self, X: NDArray, idxs: NDArray, weights=None):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _conditional_score(self.covariance, X, idxs, weights=weights)

    def conditional_score_samples(self, X: NDArray, idxs: NDArray):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _conditional_score_samples(self.covariance, X, idxs)

    def marginal_score(self, X: NDArray, idxs: NDArray, weights=None):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _marginal_score(self.covariance, X, idxs, weights=weights)

    def marginal_score_samples(self, X: NDArray, idxs: NDArray):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _marginal_score_samples(self.covariance, X, idxs)

    def score(self, X: NDArray, weights=None):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _score(self.covariance, X, weights=weights)

    def score_samples(self, X: NDArray):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _score_samples(self.covariance, X)

    #################################
    # Information-theoretic methods #
    #################################
    def entropy(self):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _entropy(self.covariance)

    def entropy_subset(self, idxs: NDArray[np.float_]):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _entropy_subset(self.covariance, idxs)

    def mutual_information(self, idxs1: NDArray[np.float_], idxs2: NDArray[np.float_]):
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

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


def _get_precision(covariance: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Gets the precision matrix defined as the inverse of the covariance

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    Returns
    -------
    precision : NDArray,(sum(p),sum(p))
        The inverse of the covariance matrix
    """
    return la.inv(covariance)


def _get_stable_rank(covariance: NDArray[np.float_]) -> np.float_:
    """
    Returns the stable rank defined as
    ||A||_F^2/||A||^2

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    Returns
    -------
    srank : np.float_
        The stable rank. See Vershynin High dimensional probability for
        discussion, but this is a statistically stable approximation to rank
    """
    singular_values = la.svd(covariance, compute_uv=False)
    return np.sum(singular_values) / singular_values[0]


def _predict(
    covariance: NDArray[np.float_], Xobs: NDArray[np.float_], idxs: NDArray[np.float_]
) -> NDArray[np.float_]:
    """
    Predicts missing data using observed data. This uses the conditional
    Gaussian formula (see wikipedia multivariate gaussian). Solve allows
    us to avoid an explicit matrix inversion.

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    Xobs : NDArray,(N_samples,\\sum idxs)
        The observed data

    idxs: NDArray,(sum(p),)
        The observation locations

    Returns
    -------
    preds : NDArray,(N_samples,p-\\sum idxs)
        The predicted values
    """
    covariance_22 = covariance[np.ix_(idxs == 1, idxs == 1)]
    covariance_21 = covariance[np.ix_(idxs == 1, idxs == 0)]
    beta_bar = la.solve(covariance_22, covariance_21)

    return np.dot(Xobs[:, idxs == 1], beta_bar)


def _conditional_score(
    covariance: NDArray[np.float_],
    X: NDArray[np.float_],
    idxs: NDArray[np.float_],
    weights: NDArray[np.float_] | None = None,
) -> np.float_:
    """
    Returns the predictive log-likelihood of a subset of data.

    mean(log p(X[idx==1]|X[idx==0],covariance))

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    X : NDArray[np.float_],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float_],(sum(p),)
        The observation locations

    weights : NDArray[np.float_],(N,),default=None
        The optional weights on the samples. Don't have negative values.
        Average value forced to 1.

    Returns
    -------
    avg_score : np.float_
        Average log likelihood
    """
    weights = np.ones(X.shape[0]) if weights is None else weights / np.mean(weights)

    return np.mean(weights * _conditional_score_samples(covariance, X, idxs))


def _conditional_score_sherman_woodbury(
    Lambda: NDArray[np.float_],
    W: NDArray[np.float_],
    X: NDArray[np.float_],
    idxs: NDArray[np.float_],
    weights: NDArray[np.float_] | None = None,
) -> np.float_:
    """
    Returns the predictive log-likelihood of a subset of data.

    mean(log p(X[idx==1]|X[idx==0],covariance))

    Parameters
    ----------
    Lambda : NDArray[np.float_],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float_],(L,p)
        The low rank component

    X : NDArray[np.float_],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float_],(sum(p),)
        The observation locations

    weights : NDArray[np.float_] | None,(N,),default=None
        The optional weights on the samples. Don't have negative values.
        Average value forced to 1.

    Returns
    -------
    avg_score : np.float_
        Average log likelihood
    """
    weights = np.ones(X.shape[0]) if weights is None else weights / np.mean(weights)

    return np.mean(
        weights * _conditional_score_samples_sherman_woodbury(Lambda, W, X, idxs)
    )


def _conditional_score_samples(
    covariance: NDArray[np.float_], X: NDArray[np.float_], idxs: NDArray[np.float_]
) -> np.float_:
    """
    Return the conditional log likelihood of each sample, that is

    log p(X[idx==1]|X[idx==0],covariance) = N(Sigma_10Sigma_00^{-1}x_0,
                                    Sigma_11-Sigma_10Sigma_00^{-1}Sigma_01)

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    X : NDArray[np.float_],(N,p)
        The centered data

    idxs: NDArray[np.float_],(p,)
        The observation locations

    Returns
    -------
    scores : np.float_
        Log likelihood for each sample
    """
    covariance_22 = covariance[np.ix_(idxs == 1, idxs == 1)]
    covariance_21 = covariance[np.ix_(idxs == 1, idxs == 0)]
    covariance_11 = covariance[np.ix_(idxs == 0, idxs == 0)]

    beta_bar = la.solve(covariance_22, covariance_21)
    Second_part = np.dot(covariance_21.T, beta_bar)

    covariance_bar = covariance_11 - Second_part

    mu_ = np.dot(X[:, idxs == 1], beta_bar)

    return _score_samples(covariance_bar, X[:, idxs == 0] - mu_)


def _conditional_score_samples_sherman_woodbury(
    Lambda: NDArray[np.float_],
    W: NDArray[np.float_],
    X: NDArray[np.float_],
    idxs: NDArray[np.float_],
) -> np.float_:
    """
    Return the conditional log likelihood of each sample, that is

    log p(X[idx==1]|X[idx==0],covariance) = N(Sigma_10Sigma_00^{-1}x_0,
                                    Sigma_11-Sigma_10Sigma_00^{-1}Sigma_01)

    Parameters
    ----------
    Lambda : NDArray[np.float_],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float_],(L,p)
        The low rank component

    X : NDArray[np.float_],(N,p)
        The centered data

    idxs: NDArray[np.float_],(p,)
        The observation locations

    Returns
    -------
    scores : np.float_
        Log likelihood for each sample
    """
    I_L = np.eye(W.shape[0])
    X_obs = X[:, idxs == 1]
    X_miss = X[:, idxs == 0]
    W_obs = W[:, idxs == 1]
    W_miss = W[:, idxs == 0]
    Lo = Lambda[np.ix_(idxs == 1, idxs == 1)]
    Lm = Lambda[np.ix_(idxs == 0, idxs == 0)]

    C = np.dot(W_obs, la.inv(Lo))
    B = np.dot(C, W_obs.T)

    IpB = I_L + B
    back = la.solve(IpB, C)
    second = C - np.dot(B, back)
    coef = np.dot(W_miss.T, second)

    mu_ = np.dot(X_obs, coef.T)
    middle = B - np.dot(B, la.solve(IpB, B))
    term2 = np.dot(W_miss.T, np.dot(middle, W_miss))

    covariance11 = Lm + np.dot(W_miss.T, W_miss)
    covariance_bar = covariance11 - term2

    return _score_samples(covariance_bar, X_miss - mu_)


def _get_conditional_parameters(
    covariance: NDArray[np.float_], idxs: NDArray[np.float_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Computes the distribution parameters p(X_miss|X_obs)

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    idxs: NDArray,(p,)
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


def _get_conditional_parameters_sherman_woodbury(
    Lambda: NDArray[np.float_], W: NDArray[np.float_], idxs: NDArray[np.float_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Computes the distribution parameters p(X_miss|X_obs)
    given that Sigma = WWT + Lambda

    Parameters
    ----------
    Lambda : NDArray,(p,p)
        The diagonal noise matrix

    W : NDArray,(L,p)
        The low rank component

    idxs: NDArray,(p,)
        The observed covariates

    Returns
    -------
    beta_bar : NDArray,(p,)
        The predictive covariates

    covariance_bar : NDArray,(p,p)
        Conditional covariance
    """
    I_L = np.eye(W.shape[0])
    W_obs = W[:, idxs == 1]
    W_miss = W[:, idxs == 0]
    Lo = Lambda[np.ix_(idxs == 1, idxs == 1)]
    Lm = Lambda[np.ix_(idxs == 0, idxs == 0)]

    C = np.dot(W_obs, la.inv(Lo))
    B = np.dot(C, W_obs.T)

    IpB = I_L + B
    back = la.solve(IpB, C)
    second = C - np.dot(B, back)
    coef = np.dot(W_miss.T, second)

    middle = B - np.dot(B, la.solve(IpB, B))
    term2 = np.dot(W_miss.T, np.dot(middle, W_miss))

    covariance11 = Lm + np.dot(W_miss.T, W_miss)
    covariance_bar = covariance11 - term2

    return coef, covariance_bar


def _marginal_score(
    covariance: NDArray[np.float_],
    X: NDArray[np.float_],
    idxs: NDArray[np.float_],
    weights: NDArray[np.float_] | None = None,
) -> np.float_:
    """
    Returns the marginal log-likelihood of a subset of data

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    X : NDArray[np.float_],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float_],(sum(p),)
        The observation locations

    weights : NDArray[np.float_],(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : np.float_
        Average log likelihood
    """
    if weights is None:
        weights = np.ones(X.shape[0])

    return np.mean(weights * _marginal_score_samples(covariance, X, idxs))


def _marginal_score_sherman_woodbury(
    Lambda: NDArray[np.float_],
    W: NDArray[np.float_],
    X: NDArray[np.float_],
    idxs: NDArray[np.float_],
    weights: NDArray[np.float_] | None = None,
) -> np.float_:
    """
    Returns the marginal log-likelihood of a subset of data
    given that Sigma = WWT + Lambda

    Parameters
    ----------
    Lambda : NDArray[np.float_],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float_],(L,p)
        The low rank component

    X : NDArray[np.float_],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float_],(sum(p),)
        The observation locations

    weights : NDArray[np.float_] | None,(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : np.float_
        Average log likelihood
    """
    if weights is None:
        weights = np.ones(X.shape[0])

    return np.mean(
        weights * _marginal_score_samples_sherman_woodbury(Lambda, W, X, idxs)
    )


def _marginal_score_samples(
    covariance: NDArray[np.float_], X: NDArray[np.float_], idxs: NDArray[np.float_]
) -> np.float_:
    """
    Returns the marginal log-likelihood of a subset of data
    per window

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    X : NDArray[np.float_],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float_],(sum(p),)
        The observation locations

    Returns
    -------
    scores : np.float_
        Average log likelihood
    """
    cov1 = covariance[idxs == 1]
    cov_sub = cov1[:, idxs == 1]

    return _score_samples(cov_sub, X)


def _marginal_score_samples_sherman_woodbury(
    Lambda: NDArray[np.float_],
    W: NDArray[np.float_],
    X: NDArray[np.float_],
    idxs: NDArray[np.float_],
) -> np.float_:
    """
    Returns the marginal log-likelihood of a subset of data
    per window given that Sigma = WWT + Lambda

    Parameters
    ----------
    Lambda : NDArray[np.float_],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float_],(L,p)
        The low rank component

    X : NDArray[np.float_],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float_],(sum(p),)
        The observation locations

    Returns
    -------
    scores : np.float_
        Average log likelihood
    """
    Lambda_sub = Lambda[np.ix_(idxs == 1, idxs == 1)]
    W_sub = W[:, idxs == 1]

    return _score_samples_sherman_woodbury(Lambda_sub, W_sub, X)


def _score(
    covariance: NDArray[np.float_],
    X: NDArray[np.float_],
    weights: NDArray[np.float_] | None = None,
) -> np.float_:
    """
    Returns the average log liklihood of data.

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    X : NDArray[np.float_],(N,sum(p))
        The centered data

    weights : NDArray[np.float_] | None,(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : np.float_
        Average log likelihood
    """
    if weights is None:
        weights = np.ones(X.shape[0])

    return np.mean(weights * _score_samples(covariance, X))


def _score_sherman_woodbury(
    Lambda: NDArray[np.float_],
    W: NDArray[np.float_],
    X: NDArray[np.float_],
    weights: NDArray[np.float_] | None = None,
) -> np.float_:
    """
    Calculate the average log likelihood of the data given a specific covariance structure.

    The covariance structure is assumed to be Sigma = WW^T + Lambda, where
    WW^T is the low-rank component and Lambda is a diagonal matrix (noise).

    Parameters
    ----------
    Lambda : NDArray[np.float_]
        The diagonal noise matrix, with shape (p, p), where 'p' represents the number of features.

    W : NDArray[np.float_]
        The low rank component matrix, with shape (L, p), where 'L' represents the number of factors
        and 'p' is the number of features.

    X : NDArray[np.float_]
        The centered data matrix, with shape (N, p), where 'N' is the number of samples and 'p' is
        the number of features. It is assumed that the data has already been centered.

    weights : NDArray[np.float_] | None, optional
        The weights for the samples, with shape (N,), where 'N' is the number of samples. If None,
        all samples are weighted equally. Default is None.

    Returns
    -------
    avg_score : np.float_
        The average log likelihood of the data under the given model parameters.

    Notes
    -----
    The function _score_samples_sherman_woodbury should be defined elsewhere and should compute
    the log likelihood of the samples given the model parameters.

    The likelihood is weighted by the `weights` parameter if provided; otherwise, each sample's
    likelihood contributes equally to the average.
    """
    if weights is None:
        weights = np.ones(X.shape[0])

    return np.mean(weights * _score_samples_sherman_woodbury(Lambda, W, X))


def _score_samples(covariance: NDArray[np.float_], X: NDArray[np.float_]) -> np.float_:
    """
    Return the log likelihood of each sample

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    X : NDArray[np.float_],(N,sum(p))
        The centered data

    Returns
    -------
    scores : np.float_
        Log likelihood for each sample
    """
    p = covariance.shape[0]
    term1 = -p / 2 * np.log(2 * np.pi)

    _, logdet = la.slogdet(covariance)
    term2 = -0.5 * logdet

    quad_init = la.solve(covariance, np.transpose(X))
    difference = X * np.transpose(quad_init)
    term3 = np.sum(difference, axis=1)

    return term1 + term2 - 0.5 * term3


def _score_samples_sherman_woodbury(
    Lambda: NDArray[np.float_], W: NDArray[np.float_], X: NDArray[np.float_]
) -> np.float_:
    """
    Return the log likelihood of each sample

    Parameters
    ----------
    Lambda : NDArray[np.float_],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float_],(L,p)
        The low rank component

    X : NDArray[np.float_],(N,sum(p))
        The centered data

    Returns
    -------
    scores : np.float_
        Log likelihood for each sample
    """
    p = Lambda.shape[1]
    I_L = np.eye(W.shape[0])
    term1 = -p / 2 * np.log(2 * np.pi)
    term2 = -0.5 * ldet_sherman_woodbury_fa(Lambda, W)

    Li = la.inv(Lambda)
    C = np.dot(Li, W.T)
    CtW = np.dot(C.T, W.T)
    middle = I_L + CtW
    end = la.solve(middle, C.T)  # (I + WLiW^T)^{-1}WLi
    prod = np.dot(C, end)
    Lip = Li - prod
    quad_init = np.dot(Lip, X.T)
    difference = X * np.transpose(quad_init)
    term3 = np.sum(difference, axis=1)

    return term1 + term2 - 0.5 * term3


def _entropy(covariance: NDArray[np.float_]) -> np.float_:
    """
    Computes the entropy of a Gaussian distribution parameterized by
    covariance.

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    Returns
    -------
    entropy : np.float_
        The differential entropy of the distribution
    """
    cov_new = 2 * np.pi * np.e * covariance
    _, logdet = la.slogdet(cov_new)

    return 0.5 * logdet


def _entropy_subset(covariance: NDArray[np.float_], idxs: NDArray[np.float_]) -> np.float_:
    """
    Computes the entropy of a subset of the Gaussian distribution
    parameterized by covariance.

    Parameters
    ----------
    covariance : NDArray[np.float_],(p,p)
        The covariance matrix

    idxs: NDArray[np.float_],(sum(p),)
        The observation locations

    Returns
    -------
    entropy : np.float_
        The differential entropy of the distribution
    """
    cov1 = covariance[idxs == 1]
    cov_sub = cov1[:, idxs == 1]
    entropy = _entropy(cov_sub)

    return entropy


def _mutual_information(
    covariance: NDArray[np.float_], idxs1: NDArray[np.float_], idxs2: NDArray[np.float_]
) -> np.float_:
    """
    This computes the mutual information bewteen the two sets of
    covariates based on the model.

    Parameters
    ----------
    idxs1 : NDArray[np.float_],(p,)
        First group of variables

    idxs2 : NDArray[np.float_],(p,)
        Second group of variables

    Returns
    -------
    mutual_information : np.float_
        The mutual information between the two variables
    """
    idxs = idxs1 + idxs2
    cov_sub = covariance[np.ix_(idxs == 1, idxs == 1)]
    idxs1_sub = idxs1[idxs == 1]
    Hy = _entropy(cov_sub[np.ix_(idxs1_sub == 1, idxs1_sub == 1)])

    _, covariance_conditional = _get_conditional_parameters(cov_sub, 1 - idxs1_sub)
    H_y_given_x = _entropy(covariance_conditional)
    mutual_information = Hy - H_y_given_x

    return mutual_information


def ldet_sherman_woodbury_fa(Lambda: NDArray, W: NDArray):
    """
    This converts the log determinant of a matrix Lambda + W^TW where
    Lambda is diagonal. Fa for factor analysis

    Parameters
    ----------
    W : np.array(n_components,p)
        The PCA loadings matrix

    Lambda : NDArray,(p,p)
        An easy to invert (diagonal) noise matrix

    Returns
    -------
    log_determinant : float
        The log determinant of the covariance matrix
    """
    _, ldetL = la.slogdet(Lambda)
    LiW = la.solve(Lambda, W.T)
    WtLiW = np.dot(W, LiW)
    IWtLiW = np.eye(W.shape[0]) + WtLiW
    _, ldetP = la.slogdet(IWtLiW)
    log_determinant = ldetP + ldetL
    return log_determinant


def ldet_sherman_woodbury_full(A: NDArray, U: NDArray, B: NDArray, V: NDArray):
    """
    The value log|A+ UBV|. Useful when A and B are simple to invert and UBV
    is low rank

    Parameters
    ----------
    A : NDArray,(p,p)
        The first square matrix

    U :  NDArray,(p,L)
        The wide matrix

    B : NDArray,(L,L)
        The tiny square matrix

    V : NDArray,(L,p)
        Tall matrix

    Returns
    -------
    log_determinant : float
        The log determinant of the covariance matrix
    """
    _, ldetA = la.slogdet(A)
    _, ldetB = la.slogdet(B)
    term2 = np.dot(V, la.solve(A, U))
    term1 = la.inv(B)
    _, ldetProd = la.slogdet(term1 + term2)
    log_determinant = ldetA + ldetB + ldetProd
    return log_determinant


def inv_sherman_woodbury_fa(Lambda: NDArray, W: NDArray):
    """
    This converts the inverse of a matrix Lambda + W^TW where
    Lambda is diagonal. Fa for factor analysis

    Parameters
    ----------
    W : np.array(n_components,p)
        The PCA loadings matrix

    Lambda : NDArray,(p,p)
        An easy to invert (diagonal) noise matrix

    Returns
    -------
    Sigma_inv : NDArray,(p,p)
        The inverse of the covariance matrix
    """
    I_L = np.eye(W.shape[0])
    I_p = np.eye(W.shape[1])
    Lambda_inv = la.inv(Lambda)
    WLi = np.dot(W, Lambda_inv)
    inner = I_L + np.dot(WLi, W.T)
    inner_inv = la.inv(inner)
    end = np.dot(inner_inv, WLi)
    term2 = np.dot(W.T, end)
    Imterm2 = I_p - term2
    Sigma_inv = np.dot(Lambda_inv, Imterm2)
    return Sigma_inv


def inv_sherman_woodbury_full(A: NDArray, U: NDArray, B: NDArray, V: NDArray):
    """
    This converts the inverse of a matrix (A + UBV)

    Parameters
    ----------
    A : NDArray,(p,p)
        The first square matrix

    U :  NDArray,(p,L)
        The wide matrix

    B : NDArray,(L,L)
        The tiny square matrix

    V : NDArray,(L,p)
        Tall matrix

    Returns
    -------
    Sigma_inv : NDArray,(p,p)
        The inverse of the matrix (A + UBV)
    """
    Ainv = la.inv(A)  # Needed explicitly anyways
    AiU = np.dot(Ainv, U)
    Binv = la.inv(B)  # Should be easy to invert
    VAinv = np.dot(V, Ainv)
    VAiU = np.dot(VAinv, U)
    middle = Binv + VAiU
    end = la.solve(middle, VAinv)
    second_term = np.dot(AiU, end)
    first_term = Ainv
    Sigma_inv = first_term - second_term
    return Sigma_inv
