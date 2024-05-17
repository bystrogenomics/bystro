import numpy as np


def operator_shrinkage(singular_values: np.ndarray, gamma: float) -> np.ndarray:
    """
    Perform operator norm optimal shrinkage on data singular values.

    Parameters
    ----------
    singular_values : array-like
        A vector of data eigenvalues, obtained from the p-by-p sample
        covariance matrix corresponding to a dataset of n samples in dimension p.
    gamma : float
        Aspect ratio p/n of the dataset.

    Returns
    -------
    shrunk_values : array
        The vector of singular values after performing optimal shrinkage.
    """
    threshold = 1 + (gamma**0.5)

    shrunk_values = []
    for sigma_k in singular_values:
        if sigma_k > threshold:
            t_k = np.sqrt(
                (
                    sigma_k**2
                    - 1
                    - gamma
                    + np.sqrt((sigma_k**2 - 1 - gamma) ** 2 - 4 * gamma)
                )
                / 2
            )
            q_k = t_k * np.sqrt(
                (t_k**2 + min(1, gamma)) / (t_k**2 + max(1, gamma))
            )
            shrunk_values.append(q_k)
        else:
            shrunk_values.append(0)

    return np.array(shrunk_values)
