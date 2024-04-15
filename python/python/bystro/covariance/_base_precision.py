"""
This provides the base class for covariance estimation in terms of the 
precision matrix. While in the future I'll rewrite separate methods for some
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
import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray


from bystro.covariance._base_covariance import BaseCovariance


class BasePrecision(BaseCovariance):
    """
    This object basically contains all the methods asides for fitting
    a precision matrix.
    """

    def __init__(self):
        super().__init__()
        self.precision: NDArray | None = None

    def _set_covariance_if_none(self):
        if self.covariance is None:
            self.covariance = self.get_covariance()

    def get_covariance(self):
        if self.precision is None:
            raise ValueError(
                "You need to fit the model first before you can get the "
                "covariance matrix"
            )

        return _get_covariance(self.precision)

    def get_stable_rank(self):
        self._set_covariance_if_none()

        return super(BasePrecision, self).get_stable_rank()

    def predict(self, Xobs: NDArray[np.float_], idxs: NDArray[np.float_]):
        self._set_covariance_if_none()

        return super(BasePrecision, self).predict(Xobs, idxs)

    def conditional_score(
        self,
        X: NDArray[np.float_],
        idxs: NDArray[np.float_],
        weights: NDArray[np.float_] | None = None,
    ):
        self._set_covariance_if_none()

        return super(BasePrecision, self).conditional_score(
            X, idxs, weights=weights
        )

    def conditional_score_samples(
        self, X: NDArray[np.float_], idxs: NDArray[np.float_]
    ):
        self._set_covariance_if_none()

        return super(BasePrecision, self).conditional_score_samples(X, idxs)

    def marginal_score(
        self,
        X: NDArray[np.float_],
        idxs: NDArray[np.float_],
        weights: NDArray[np.float_] | None = None,
    ):
        self._set_covariance_if_none()

        return super(BasePrecision, self).marginal_score(
            X, idxs, weights=weights
        )

    def marginal_score_samples(
        self, X: NDArray[np.float_], idxs: NDArray[np.float_]
    ):
        self._set_covariance_if_none()

        return super(BasePrecision, self).marginal_score_samples(X, idxs)

    def score(
        self, X: NDArray[np.float_], weights: NDArray[np.float_] | None = None
    ):
        self._set_covariance_if_none()

        return super(BasePrecision, self).score(X, weights=weights)

    def score_samples(self, X: NDArray[np.float_]):
        self._set_covariance_if_none()

        return super(BasePrecision, self).score_samples(X)

    def entropy(self):
        self._set_covariance_if_none()

        return super(BasePrecision, self).entropy()

    def entropy_subset(self, idxs: NDArray[np.float_]):
        self._set_covariance_if_none()

        return super(BasePrecision, self).entropy_subset(idxs)

    def mutual_information(
        self, idxs1: NDArray[np.float_], idxs2: NDArray[np.float_]
    ):
        self._set_covariance_if_none()

        return super(BasePrecision, self).mutual_information(idxs1, idxs2)


def _get_covariance(precision: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Gets the covariance matrix defined as the inverse of the precision

    Parameters
    ----------
    precision : NDArray,(p,p)
        The precision matrix

    Returns
    -------
    covariance : NDArray,(sum(p),sum(p))
        The inverse of the precision matrix
    """
    return la.inv(precision)
