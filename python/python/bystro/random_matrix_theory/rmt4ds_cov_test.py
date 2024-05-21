import numpy as np
import numpy.linalg as la
from scipy.stats import norm # type: ignore
from numpy.linalg import det, solve


def d2(y1, y2):
    """
    Compute a specific mathematical formula involving the logarithm and
    division operations.

    Parameters
    ----------
    y1 : float
        The first input value used in the computation.
    y2 : float
        The second input value used in the computation.

    Returns
    -------
    float
        The result of the computed formula based on y1 and y2.
    """
    return (
        (y1 + y2 - y1 * y2)
        / (y1 * y2)
        * np.log((y1 + y2) / (y1 + y2 - y1 * y2))
        + y1 * (1 - y2) / (y2 * (y1 + y2)) * np.log(1 - y2)
        + y2 * (1 - y1) / (y1 * (y1 + y2)) * np.log(1 - y1)
    )


def mu2(y1, y2):
    """
    Compute a logarithmic formula used in statistical calculations involving covariance.

    Parameters
    ----------
    y1 : float
        The first input value.
    y2 : float
        The second input value.

    Returns
    -------
    float
        The result of the computed formula.
    """
    return 0.5 * np.log((y1 + y2 - y1 * y2) / (y1 + y2)) - (
        y1 * np.log(1 - y2) + y2 * np.log(1 - y1)
    ) / (y1 + y2)


def sigma2_2(y1, y2):
    """
    Compute the variance component used in statistical tests for comparing covariances.

    Parameters
    ----------
    y1 : float
        The first input value.
    y2 : float
        The second input value.

    Returns
    -------
    float
        The computed variance based on y1 and y2.
    """
    return -(2 * y1**2 * np.log(1 - y2) + 2 * y2**2 * np.log(1 - y1)) / (
        y1 + y2
    ) ** 2 - 2 * np.log((y1 + y2) / (y1 + y2 - y1 * y2))


def d2_hat(y1, y2):
    """
    Adjust the d2 computation by accounting for the proportions of y1 and y2 in the total.

    Parameters
    ----------
    y1 : float
        The first input value.
    y2 : float
        The second input value.

    Returns
    -------
    float
        The adjusted result of the d2 formula.
    """
    return (
        d2(y1, y2)
        - y1 / (y1 + y2) * np.log(y1 / (y1 + y2))
        - y2 / (y1 + y2) * np.log(y2 / (y1 + y2))
    )


def one_sample_cov_test(X, mean=None, S=None):
    """
    Perform a one-sample covariance test to evaluate if the sample
    covariance differs
    from the identity matrix or another specified covariance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data matrix.
    mean : ndarray of shape (n_features,), default=None
        The mean vector for adjusting the data. If None, use the sample mean.
    S : ndarray of shape (n_features, n_features), default=None
        The covariance matrix for the test. If None, the identity matrix is assumed.

    Returns
    -------
    dict
        A dictionary containing:
        - 'p_value': The p-value of the test.
        - 'z_value': The computed Z-value for the test.
        - 'lrt': The likelihood ratio test statistic.
    """
    n, p = X.shape
    y = p / n
    N = n - 1
    yN = p / N

    if S is not None:
        S_half = la.cholesky(S)
        X = X @ la.inv(S_half)

    if mean is None:
        X = X - np.mean(X, axis=0)
        S = X.T @ X / N
    else:
        X = X - mean
        S = X.T @ X / n

    lrt = np.sum(np.diag(S)) - np.log(det(S)) - p
    mu1 = -0.5 * np.log(1 - y)
    sigma1 = -2 * np.log(1 - y) - 2 * y
    z_value = (lrt - p * (1 + (1 - yN) / yN * np.log(1 - yN)) - mu1) / np.sqrt(
        sigma1
    )
    p_value = norm.sf(z_value)

    return {"p_value": p_value, "z_value": z_value, "lrt": lrt}


def two_sample_cov_test(X1, X2, mean=None):
    """
    Perform a two-sample covariance test to determine if there is a significant
    difference between the covariances of two samples.

    Parameters
    ----------
    X1 : ndarray of shape (n_samples1, n_features)
        The first input data matrix.
    X2 : ndarray of shape (n_samples2, n_features)
        The second input data matrix.
    mean : ndarray of shape (n_features,), default=None
        The mean vector to be used for adjusting both samples. If None, use the sample means.

    Returns
    -------
    dict
        A dictionary containing:
        - 'p_value': The p-value of the test.
        - 'z_value': The computed Z-value for the test.
        - 'lrt': The likelihood ratio test statistic.
    """
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("Input data should have the same dimension")

    n1, p1 = X1.shape
    n2, p2 = X2.shape
    if mean is not None:
        N1 = n1
        N2 = n2
        X1 = X1 - mean
        X2 = X2 - mean
    else:
        N1 = n1 - 1
        N2 = n2 - 1
        X1 = X1 - np.mean(X1, axis=0)
        X2 = X2 - np.mean(X2, axis=0)

    N = N1 + N2
    c1 = N1 / N
    c2 = N2 / N
    yN1 = p1 / N1
    yN2 = p2 / N2
    S1 = X1.T @ X1 / N1
    S2 = X2.T @ X2 / N2

    log_V1_ = np.log(det(S1 @ solve(S2, np.eye(p1)))) * (N1 / 2) - np.log(
        det(c1 * S1 @ solve(S2, np.eye(p2)) + c2 * np.eye(p2))
    ) * (N / 2)
    z_value = (-2 / N * log_V1_ - p1 * d2(yN1, yN2) - mu2(yN1, yN2)) / np.sqrt(
        sigma2_2(yN1, yN2)
    )
    p_value = norm.sf(z_value)

    return {"p_value": p_value, "z_value": z_value, "lrt": -2 * log_V1_}


def multi_sample_cov_test(*args, input_data=None):
    """
    Perform a multiple-sample covariance test to evaluate if there are significant
    differences among the covariances of multiple samples.

    Parameters
    ----------
    *args : sequence of ndarrays
        The input data matrices as variadic arguments.
    input : list of ndarrays, default=None
        The input data matrices provided as a list.

    Returns
    -------
    dict
        A dictionary containing:
        - 'p_value': The p-value of the test.
        - 'z_value': The computed Z-value for the test.
        - 'lrt': The likelihood ratio test statistic.

    """
    matrices = list(args) if input_data is None else input_data

    ps = [m.shape[1] for m in matrices]
    Ns = [m.shape[0] - 1 for m in matrices]
    q = len(matrices)

    if len(set(ps)) != 1:
        raise ValueError("Input data should have the same dimension")

    p = ps[0]
    As = np.zeros((q, p, p))

    for i in range(q):
        As[i, :, :] = np.cov(matrices[i], rowvar=False) * Ns[i]

    y_n1 = []
    y_n2 = []
    logV1 = []

    for i in range(1, q):
        y_n1.append(p / sum(Ns[:i]))
        y_n2.append(p / Ns[i])

        if i == 1:
            logV1.append(
                Ns[0] / 2 * np.log(det(As[0, :, :]))
                + Ns[1] / 2 * np.log(det(As[1, :, :]))
                - sum(Ns[:2]) / 2 * np.log(det(np.sum(As[:2, :, :], axis=0)))
            )
        else:
            logV1.append(
                sum(Ns[:i]) / 2 * np.log(det(np.sum(As[:i, :, :], axis=0)))
                + Ns[i] / 2 * np.log(det(As[i, :, :]))
                - sum(Ns[: i + 1])
                / 2
                * np.log(det(np.sum(As[: i + 1, :, :], axis=0)))
            )

    test = 0
    sigma2 = 0
    mu = 0
    lrt = 0

    for i in range(1, q):
        test -= 2 / sum(Ns[: i + 1]) * logV1[i - 1] - p * d2_hat(
            y_n1[i - 1], y_n2[i - 1]
        )
        lrt -= 2 / sum(Ns[: i + 1]) * logV1[i - 1]
        mu += mu2(y_n1[i - 1], y_n2[i - 1])
        sigma2 += sigma2_2(y_n1[i - 1], y_n2[i - 1])

    z_value = (test - mu) / np.sqrt(sigma2)
    p_value = norm.sf(z_value)

    return {"p_value": p_value, "z_value": z_value, "lrt": lrt}
