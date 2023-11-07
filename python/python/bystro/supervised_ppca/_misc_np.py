"""

Objects
-------

Methods
-------

"""
import numpy as np


def softplus_inverse_np(y):
    """
    Computes the inverse of the softplus activation of x in a
    numerically stable way

    Softplus: y = log(exp(x) + 1)
    Softplus^{-1}: y = np.log(np.exp(x) - 1)

    Parameters
    ----------
    x : np.array
        Original array

    Returns
    -------
    x : np.array
        Transformed array
    """
    min_threshold = 10**-15
    max_threshold = 500
    safe_y = np.clip(
        y, min_threshold, max_threshold
    )  # we can safely pass this to the reference inverse_softplus below
    safe_x = np.log(np.exp(safe_y) - 1)

    # if y_i was below (respectively: above) the min (max) threshold, replace with log(y_i)  (y_i)
    x = np.where(y < min_threshold, np.log(y), safe_x)
    x = np.where(y > max_threshold, y, x)
    return x


def subset_square_matrix_np(Sigma, idxs):
    """
    This returns a symmetric subset of a square matrix

    Parameters
    ----------
    Sigma : np.array-like,(p,p)
        Covariance matrix (presumably)

    idxs : np.array-like,(p,)
        Binary vector, 1 means select
    Returns
    -------
    Sigma_sub : np.array-like,(sum(idxs),sum(idxs))
        The subset of matrix
    """
    Sigma_sub = Sigma[np.ix_(idxs == 1, idxs == 1)]
    return Sigma_sub
