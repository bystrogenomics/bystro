import numpy as np
from numpy.linalg import inv


def opt_shrink(singular_values: np.ndarray, r: int) -> np.ndarray:
    """
    Perform operator norm optimal shrinkage on data singular values.

    Parameters
    ----------
    singular_values : array-like
        A vector of data eigenvalues, obtained by running eig on the p-by-p sample covariance matrix
        corresponding to a dataset of n samples in dimension p.
    r : int
        Estimate of the effective rank of the latent low-rank signal matrix
        singular values.

    Returns
    -------
    shrunk_values : np.ndarray
        The vector of singular values after performing optimal shrinkage.
    """
    Sigma2_r = np.square(singular_values[r:])

    shrunk_values = []
    n = len(Sigma2_r)

    for z_k in singular_values[:r]:
        resolvent = inv(z_k**2 * np.eye(n) - np.diag(Sigma2_r))
        trace_resolvent = np.trace(z_k * resolvent)
        trace_resolvent_square = np.trace(-2 * z_k**2 * resolvent @ resolvent + resolvent)
        D = (trace_resolvent / n) ** 2
        D_prime = 2 / n * (trace_resolvent * trace_resolvent_square)
        w_opt = -2 * D / D_prime
        shrunk_values.append(w_opt)

    return np.array(shrunk_values)
