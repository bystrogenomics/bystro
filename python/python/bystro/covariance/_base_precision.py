"""
This provides the base class for covariance estimation in terms of the 
precision matrix. While in the futre I'll rewrite separate methods for some
of these to take advantage of the fact that we estimated the precision 
matrix rather than covariance matrix, for now I just invert precision and 
use the covariance matrix methods. Yuck right?

Objects
-------
BasePrecision

    get_covariance()
        Gets the precision matrix defined as the inverse of the covariance 

    get_stable_rank()
        Returns the stable rank defined as 
        ||A||_F^2/||A||^2

    predict(Xobs,idxs)
        Predicts missing data using observed data.

    --------------------
    conditional_score(X,idxs)
        mean(log p(X[idx==1]|X[idx==0],covariance))

    conditional_score_samples(X,idxs)
        log p(X[idx==1]|X[idx==0],covariance)
    
    marginal_score(X,idxs)
        mean(log p(X[idx==1]|covariance))

    marginal_score_samples(X,idxs)
        log p(X[idx==1]|covariance)

    score(X):
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
_get_covariance(covariance)
"""
from datetime import datetime as dt

from numpy import linalg as la
import pytz

from bystro.covariance._base_covariance import BaseCovariance


class BasePrecision(BaseCovariance):
    """
    This object basically contains all the methods asides for fitting
    a precision matrix.
    """

    def __init__(self):
        self.creationDate = dt.now(pytz.timezone("US/Pacific"))

    #################
    # Miscellaneous #
    #################
    def get_covariance(self):
        return _get_covariance(self.precision)

    def get_stable_rank(self):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).get_stable_rank()

    def predict(self, Xobs, idxs):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).predict(Xobs, idxs)

    #####################################
    # Gaussian Likelihood-based methods #
    #####################################
    def conditional_score(self, X, idxs, weights=None):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).conditional_score(
            X, idxs, weights=weights
        )

    def conditional_score_samples(self, X, idxs):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).conditional_score_samples(X, idxs)

    def marginal_score(self, X, idxs, weights=None):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).marginal_score(
            X, idxs, weights=weights
        )

    def marginal_score_samples(self, X, idxs):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).marginal_score_samples(X, idxs)

    def score(self, X, weights=None):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).score(X, weights=weights)

    def score_samples(self, X):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).score_samples(X)

    #################################
    # Information-theoretic methods #
    #################################
    def entropy(self):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).entropy()

    def entropy_subset(self, idxs):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).entropy_subset(idxs)

    def mutual_information(self, idxs1, idxs2):
        if not hasattr(self, "covariance"):
            self.covariance = la.inv(self.precision)
        return super(BasePrecision, self).mutual_information(idxs1, idxs2)


def _get_covariance(precision):
    """
    Gets the covariance matrix defined as the inverse of the precision

    Parameters
    ----------
    precision : np.array-like(p,p)
        The precision matrix

    Returns
    -------
    covariance : np.array-like(sum(p),sum(p))
        The inverse of the precision matrix
    """
    covariance = la.inv(precision)
    return covariance
