"""

Objects
-------

Methods
-------

"""
import numpy as np


def softplus_inverse_np(x):
    """
    Computes the inverse of the softplus activation of x in a
    numerically stable way
    y = np.log(np.exp(x) - 1)

    Parameters
    ----------
    x : np.array
        Original array

    Returns
    -------
    x : np.array
        Transformed array
    """
    threshold = np.log(np.finfo(x.dtype).eps) + 2.0
    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = np.log(x)
    too_large_value = x
    y = x + np.log(-(np.exp(-x) - 1))
    y[is_too_small] = too_small_value[is_too_small]
    y[is_too_large] = too_large_value[is_too_large]
    return y


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
