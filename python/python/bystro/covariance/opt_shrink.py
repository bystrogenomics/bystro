import numpy as np
import numpy.linalg as la


def opt_shrink(singular_values, r):
    """
    Perform operator norm optimal shrinkage on data singular values.

    Parameters
    ----------
    singular_values : array-like
        A vector of data eigenvalues, obtained by running eig on the p-by-p sample covariance matrix
        corresponding to a dataset of n samples in dimension p.
    r : float
        Estimate of the effective rank of the latent low-rank signal matrix
        singular values

    Returns
    -------
    shrunk_values : array
        The vector of singular values after performing optimal shrinkage.

    """
    Sigma2_r = np.diag(singular_values[r:]) ** 2
    n = len(Sigma2_r)

    shrunk_values = []
    for z_k in singular_values[:r]:
        resolvent = la.inv(z_k**2 * np.eye(n) - Sigma2_r)
        D = 1 / n**2 * np.trace(z_k * resolvent) ** 2
        D_prime = (
            2
            / n**2
            * (np.trace(z_k * resolvent) * np.trace(-2 * z_k**2 * resolvent**2 + resolvent))
        )
        w_opt = -2 * D / D_prime
        shrunk_values.append(w_opt)
    return np.array(shrunk_values)
