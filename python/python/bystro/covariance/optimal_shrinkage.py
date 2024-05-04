import numpy as np
from scipy.integrate import quad
from typing import Callable


def optimal_shrinkage(
    eigenvals: np.ndarray, gamma: float, loss: str = "N_1", sigma: float = None
) -> tuple[np.ndarray, float]:
    """
    Perform optimal shrinkage on data eigenvalues under various loss functions.

    Parameters
    ----------
    eigenvals : array-like
        A vector of data eigenvalues, obtained by running eig on the p-by-p sample covariance matrix
        corresponding to a dataset of n samples in dimension p.
    gamma : float
        Aspect ratio p/n of the dataset. For convenience, we assume p <= n.
    loss : str
        The loss function for which the shrinkage should be optimal. Available options:
            - 'F_1': Frobenius norm on A-B
            - 'F_2': Frobenius norm on precision (A^-1 - B^-1)
            - 'F_3': Frobenius norm on A^-1 * B - I
            - 'F_4': Frobenius norm on B^-1 * A - I
            - 'F_6': Frobenius norm on A^1/2 * B * A^1/2 - I
            - 'N_1': Nuclear norm on A-B
            - 'N_2': Nuclear norm on precision (A^-1 - B^-1)
            - 'N_3': Nuclear norm on A^-1 * B - I
            - 'N_4': Nuclear norm on B^-1 * A - I
            - 'N_6': Nuclear norm on A^1/2 * B * A^1/2 - I
            - 'O_1': Operator norm on A-B
            - 'O_2': Operator norm on precision (A^-1 - B^-1)
            - 'O_6': Operator norm on A^1/2 * B * A^1/2 - I
            - 'Stein': Stein's loss
            - 'Ent': Entropy loss
            - 'Div': Divergence loss
            - 'Fre': Frechet loss
            - 'Aff': Affine loss
    sigma : float or None, optional
        Noise standard deviation (of each entry of the noise matrix). If not provided, the noise level
        is estimated from the data.

    Returns
    -------
    eigenvals : array
        The vector of eigenvalues after performing optimal shrinkage.
    sigma : float
        An estimate of the noise level.

    Usage
    -----
    Given an n-by-p data matrix Y assumed to follow approximately the spiked model (that is, each row
    is an i.i.d sample from a p-variate Gaussian distribution, whose population covariance is a multiple
    of the identity except for a small number of top population "spikes"), we form an estimate of the
    population covariance as follows:

        [n, p] = Y.shape
        S = Y.T @ Y
        V, D = np.linalg.eig(S)
        d = np.diag(D)
        d = optimal_shrinkage(d, p/n, 'F_1')
        Shat = V @ np.diag(d) @ V.T

    Replace 'F_1' with one of the other losses. If the noise level sigma is known, use:

        d = optimal_shrinkage(d, p/n, 'F_1', sigma)

    """
    assert np.prod(np.shape(gamma)) == 1
    assert gamma <= 1
    assert gamma > 0
    assert np.prod(np.shape(eigenvals)) == len(eigenvals)

    # Estimate sigma if needed
    if sigma is None:
        MPmedian = median_marcenko_pastur(gamma)
        sigma = np.sqrt(np.median(eigenvals) / MPmedian)

    sigma2 = sigma**2
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    eigenvals_new = eigenvals / sigma2
    ind = eigenvals_new > lam_plus
    eigenvals_new[~ind] = 1

    ell_vals = ell(eigenvals_new[ind], gamma)
    c_vals = c(eigenvals_new[ind], gamma)
    s_vals = s(eigenvals_new[ind], gamma)

    if loss == "F_1":
        shrunk_vals = shrinkage_frobenius_1(ell_vals, c_vals)
    elif loss == "F_2":
        shrunk_vals = shrinkage_frobenius_2(ell_vals, c_vals, s_vals)
    elif loss == "F_3":
        shrunk_vals = shrinkage_frobenius_3(ell_vals, c_vals, s_vals)
    elif loss == "F_4":
        shrunk_vals = shrinkage_frobenius_4(ell_vals, c_vals, s_vals)
    elif loss == "F_6":
        shrunk_vals = shrinkage_frobenius_6(ell_vals, c_vals, s_vals)
    elif loss == "N_1":
        shrunk_vals = shrinkage_nuclear_1(ell_vals, s_vals)
    elif loss == "N_2":
        shrunk_vals = shrinkage_nuclear_2(ell_vals, c_vals, s_vals)
    elif loss == "N_3":
        shrunk_vals = shrinkage_nuclear_3(ell_vals, c_vals, s_vals)
    elif loss == "N_4":
        shrunk_vals = shrinkage_nuclear_4(ell_vals, c_vals, s_vals)
    elif loss == "N_6":
        shrunk_vals = shrinkage_nuclear_6(ell_vals, c_vals, s_vals)
    elif loss == "O_1":
        shrunk_vals = shrinkage_operator_1(ell_vals)
    elif loss == "O_2":
        shrunk_vals = shrinkage_operator_2(ell_vals)
    elif loss == "O_6":
        shrunk_vals = shrinkage_operator_6(ell_vals, c_vals, s_vals)
    elif loss == "Stein":
        shrunk_vals = shrinkage_stein(ell_vals, c_vals, s_vals)
    elif loss == "Ent":
        shrunk_vals = shrinkage_entropy(ell_vals, c_vals, s_vals)
    elif loss == "Div":
        shrunk_vals = shrinkage_divergence(ell_vals, c_vals, s_vals)
    elif loss == "Fre":
        shrunk_vals = shrinkage_frechet(ell_vals, c_vals, s_vals)
    elif loss == "Aff":
        shrunk_vals = shrinkage_affine(ell_vals, c_vals, s_vals)
    else:
        raise ValueError("Unrecognized shrinkage %s" % loss)

    eigenvals_new[ind] = shrunk_vals
    eigenvals_new *= sigma2

    return eigenvals_new, sigma


def ell(lam: np.ndarray, gamma: float) -> np.ndarray:
    """Calculate a transformation of lambda with parameter gamma."""
    term = lam + 1 - gamma
    return (term + np.sqrt(term**2 - 4 * lam)) / 2


def c(lam: np.ndarray, gamma: float) -> np.ndarray:
    """Calculate the c component for shrinkage based on lambda."""
    ell_val = ell(lam, gamma)
    return np.sqrt((1 - gamma / ((ell_val - 1) ** 2)) / (1 + gamma / (ell_val - 1)))


def s(lam: np.ndarray, gamma: float) -> np.ndarray:
    """Calculate the s component for shrinkage based on lambda."""
    c_val = c(lam, gamma)
    return np.sqrt(1 - c_val**2)


def shrinkage_frobenius_1(ell: np.ndarray, c: np.ndarray) -> np.ndarray:
    """max(1 + (c^2) * (ell - 1), 0)"""
    return np.maximum(1 + (c**2) * (ell - 1), 0)


def shrinkage_frobenius_2(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """max(ell / ((c^2) + ell * (s^2)), 0)"""
    return np.maximum(ell / ((c**2) + ell * (s**2)), 0)


def shrinkage_frobenius_3(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """max(1 + (ell - 1) * ((c^2) / ((ell^2) * (s^2) + c^2)), 1)"""
    return np.maximum(1 + (ell - 1) * ((c**2) / ((ell**2) * (s**2) + c**2)), 1)


def shrinkage_frobenius_4(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(s^2 + (ell^2) * (c^2)) / ((s^2) + (ell * (c^2)))"""
    return (s**2 + (ell**2) * (c**2)) / ((s**2) + (ell * (c**2)))


def shrinkage_frobenius_6(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """1 + ((ell - 1) * (c^2)) / (((c^2) + ell * (s^2))^2)"""
    return 1 + ((ell - 1) * (c**2)) / (((c**2) + ell * (s**2)) ** 2)


def shrinkage_operator_1(ell: np.ndarray) -> np.ndarray:
    """Debiasing to population eigenvalues"""
    return ell


def shrinkage_operator_2(ell: np.ndarray) -> np.ndarray:
    """Debiasing to population eigenvalues"""
    return ell


def shrinkage_operator_6(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """1 + ((ell - 1) / (c^2 + ell * (s^2)))"""
    return 1 + ((ell - 1) / (c**2 + ell * (s**2)))


def shrinkage_nuclear_1(ell: np.ndarray, s: np.ndarray) -> np.ndarray:
    """max(1 + (ell - 1) * (1 - 2 * (s^2)), 1)"""
    return np.maximum(1 + (ell - 1) * (1 - 2 * (s**2)), 1)


def shrinkage_nuclear_2(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """max(ell / ((2 * ell - 1) * (s^2) + c^2), 1)"""
    return np.maximum(ell / ((2 * ell - 1) * (s**2) + c**2), 1)


def shrinkage_nuclear_3(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """max(ell / (c^2 + (ell^2) * (s^2)), 1)"""
    return np.maximum(ell / (c**2 + (ell**2) * (s**2)), 1)


def shrinkage_nuclear_4(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """max(ell * (c^2) + (s^2) / ell, 1)"""
    return np.maximum(ell * (c**2) + (s**2) / ell, 1)


def shrinkage_nuclear_6(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """max((ell - ((ell - 1)^2) * (c^2) * (s^2)) / (((c^2) + ell * (s^2))^2), 1)"""
    return np.maximum(
        (ell - ((ell - 1) ** 2) * (c**2) * (s**2)) / (((c**2) + ell * (s**2)) ** 2),
        1,
    )


def shrinkage_stein(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ell / (c^2 + ell * (s^2))"""
    return ell / (c**2 + ell * (s**2))


def shrinkage_entropy(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ell * (c^2) + s^2"""
    return ell * (c**2) + s**2


def shrinkage_divergence(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """sqrt(((ell^2) * (c^2) + ell * (s^2)) / (c^2 + (s^2) * ell))"""
    return np.sqrt(((ell**2) * (c**2) + ell * (s**2)) / (c**2 + (s**2) * ell))


def shrinkage_frechet(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """((sqrt(ell) * (c^2) + s^2)^2)"""
    return (np.sqrt(ell) * (c**2) + s**2) ** 2


def shrinkage_affine(ell: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """((1 + c^2) * ell + (s^2)) / (1 + (c^2) + ell * (s^2))"""
    return ((1 + c**2) * ell + (s**2)) / (1 + (c**2) + ell * (s**2))


def marcenko_pastur_integral(x: float, gamma: float) -> float:
    """
    Compute the integral of the Marcenko-Pastur distribution.

    Parameters
    ----------
    x : float
        Upper limit of integration.
    gamma : float
        Parameter of the distribution.

    Returns
    -------
    integral : float
        The computed integral of the Marcenko-Pastur distribution.

    Compute the integral of the Marcenko-Pastur distribution up to the given upper limit.
    """
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma beyond")
    lobnd = (1 - np.sqrt(gamma)) ** 2
    hibnd = (1 + np.sqrt(gamma)) ** 2
    if (x < lobnd) or (x > hibnd):
        raise ValueError("x beyond")

    def dens(t: np.ndarray, gamma: float, lobnd: float, hibnd: float) -> np.ndarray:
        """
        Compute the Marcenko-Pastur density function.

        Parameters
        ----------
        t : array_like
            Input values.
        gamma : float
            Parameter of the Marcenko-Pastur distribution.
        lobnd : float
            Lower bound of the distribution.
        hibnd : float
            Upper bound of the distribution.

        Returns
        -------
        density : array_like
            Marcenko-Pastur density function values corresponding to input values.

        This function computes the Marcenko-Pastur density function.
        """
        density = np.sqrt((hibnd - t) * (t - lobnd)) / (2 * np.pi * gamma * t)
        return density

    integral, _ = quad(dens, lobnd, x)
    return integral


def median_marcenko_pastur(gamma: float) -> float:
    """
    Compute the median of the Marcenko-Pastur distribution.

    Parameters
    ----------
    gamma : float
        Parameter of the Marcenko-Pastur distribution.

    Returns
    -------
    med : float
        The computed median of the Marcenko-Pastur distribution.

    This function iteratively computes the median of the Marcenko-Pastur distribution
    using a binary search algorithm.
    """

    def mar_pas_ccdf(x: float) -> float:
        """
        Compute the complementary cumulative distribution function (CCDF) of the Marcenko-Pastur
        distribution.

        Parameters
        ----------
        x : float
            Input value.

        Returns
        -------
        ccdf : float
            Complementary cumulative distribution function (1 - CDF) of the Marcenko-Pastur
            distribution corresponding to input value.

        This function computes the complementary cumulative distribution function (CCDF) of
        the Marcenko-Pastur distribution by subtracting the cumulative distribution function
        (CDF) computed by the `inc_mar_pas` function from 1.
        """
        ccdf = 1 - inc_mar_pas(x, gamma, 0)
        return ccdf

    lobnd = (1 - np.sqrt(gamma)) ** 2
    hibnd = (1 + np.sqrt(gamma)) ** 2
    change = True
    while change and (hibnd - lobnd > 0.001):
        change = False
        x = np.linspace(lobnd, hibnd, 5)
        y = np.array([mar_pas_ccdf(xi) for xi in x])
        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = True
        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = True

    med = (hibnd + lobnd) / 2
    return med


def inc_mar_pas(x0: float, gamma: float, alpha: float) -> float:
    """
    Compute the integral of the Marcenko-Pastur distribution.

    Parameters
    ----------
    x0 : float
        Lower limit of integration.
    gamma : float
        Parameter of the Marcenko-Pastur distribution.
    alpha : float
        Exponent of the power function.

    Returns
    -------
    integral : float
        The computed integral of the Marcenko-Pastur distribution.

    This function computes the integral of the Marcenko-Pastur distribution using numerical integration.
    """
    if gamma > 1:
        raise ValueError("gammaBeyond")

    top_spec = (1 + np.sqrt(gamma)) ** 2
    bot_spec = (1 - np.sqrt(gamma)) ** 2

    def if_else(Q: np.ndarray, point: np.ndarray, counter_point: np.ndarray) -> np.ndarray:
        """
        Choose between two values based on a condition.

        Parameters
        ----------
        Q : array_like
            Boolean condition array.
        point : array_like
            Values to choose when the condition is True.
        counter_point : array_like or float
            Values to choose when the condition is False.

        Returns
        -------
        y : array_like
            Selected values based on the condition.

        This function selects values from `point` or `counter_point` based on the boolean condition `Q`.
        """
        y = point.copy()
        if np.any(~Q):
            if len(counter_point) == 1:
                counter_point = np.ones_like(Q) * counter_point
            y[~Q] = counter_point[~Q]
        return y

    def mar_pas(x: np.ndarray) -> np.ndarray:
        """
        Compute the Marcenko-Pastur distribution.

        Parameters
        ----------
        x : array_like
            Input values.

        Returns
        -------
        dist: array_like
            Marcenko-Pastur distribution values corresponding to input values.

        This function computes the Marcenko-Pastur distribution using the if_else function.
        """
        dist = if_else(
            (top_spec - x) * (x - bot_spec) > 0,
            np.sqrt((top_spec - x) * (x - bot_spec)) / (gamma * x) / (2 * np.pi),
            np.array([0]),
        )
        return dist

    def fun(x: np.ndarray, alpha: float, mar_pas: Callable) -> np.ndarray:
        """
        Compute the Marcenko-Pastur function.

        Parameters
        ----------
        x : array_like
            Input values.
        alpha : float
            Exponent of the power function.
        mar_pas : callable
            Function computing the Marcenko-Pastur distribution.

        Returns
        -------
        array_like
            Marcenko-Pastur function values corresponding to input values.

        This function computes the Marcenko-Pastur function.
        """
        return x**alpha * mar_pas(x)

    # Numerical integration using the trapezoidal rule
    x_values = np.linspace(x0, top_spec, 1000)
    integral = np.trapz(fun(x_values, alpha, mar_pas), x_values)
    return integral
