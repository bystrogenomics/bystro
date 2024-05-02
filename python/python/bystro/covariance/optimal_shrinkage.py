import numpy as np
from scipy.integrate import quad


def optimal_shrinkage(eigenvals, gamma, loss="N_1", sigma=None):
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
        MPmedian = MedianMarcenkoPastur(gamma)
        sigma = np.sqrt(np.median(eigenvals) / MPmedian)

        int(f"estimated sigma={sigma:.2f}")

    sigma2 = sigma**2
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    eigenvals_new = eigenvals / sigma2
    I = eigenvals_new > lam_plus
    eigenvals_new[~I] = 1

    ell_vals = ell()
    c_vals = c()
    s_vals = np.sqrt(1 - c(lam, lam_plus, gamma) ** 2)

    if loss == "F_1":
        shrunk_vals = shrinkage_frobenius_1(ell_vals, c_vals, s_vals)
    elif loss == "F_2":
        shrunk_vals = shrinkage_frobenius_2(ell_vals, c_vals, s_vals)
    elif loss == "F_3":
        shrunk_vals = shrinkage_frobenius_3(ell_vals, c_vals, s_vals)
    elif loss == "F_4":
        shrunk_vals = shrinkage_frobenius_4(ell_vals, c_vals, s_vals)
    elif loss == "F_6":
        shrunk_vals = shrinkage_frobenius_6(ell_vals, c_vals, s_vals)
    elif loss == "N_1":
        shrunk_vals = shrinkage_nuclear_1(ell_vals, c_vals, s_vals)
    elif loss == "N_2":
        shrunk_vals = shrinkage_nuclear_2(ell_vals, c_vals, s_vals)
    elif loss == "N_3":
        shrunk_vals = shrinkage_nuclear_3(ell_vals, c_vals, s_vals)
    elif loss == "N_5":
        shrunk_vals = shrinkage_nuclear_5(ell_vals, c_vals, s_vals)
    elif loss == "N_6":
        shrunk_vals = shrinkage_nuclear_6(ell_vals, c_vals, s_vals)
    elif loss == "O_1":
        shrunk_vals = shrinkage_operator_1(ell_vals, c_vals, s_vals)
    elif loss == "O_2":
        shrunk_vals = shrinkage_operator_2(ell_vals, c_vals, s_vals)
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

    eigenvals_new[I] = shrunk_vals
    eigenvals_new *= sigma2

    eigenvals = optshrink_impl(eigenvals, gamma, loss, sigma)
    return eigenvals, sigma


def ell(lam, lam_plus, gamma):
    return np.where(
        lam >= lam_plus,
        ((lam + 1 - gamma) + np.sqrt((lam + 1 - gamma) ** 2 - 4 * lam)) / 2.0,
        0,
    )


def c(lam, lam_plus, gamma):
    return np.where(
        lam >= lam_plus,
        np.sqrt(
            (1 - gamma / ((ell(lam, lam_plus, gamma) - 1) ** 2))
            / (1 + gamma / (ell(lam, lam_plus, gamma) - 1))
        ),
        0,
    )


def s(lam, lam_plus, gamma):
    return np.sqrt(1 - c(lam, lam_plus, gamma) ** 2)


def shrinkage_frobenius_1(ell, c, s):
    """max(1 + (c^2) * (ell - 1), 0)"""
    return np.maximum(1 + (c**2) * (ell - 1), 0)


def shrinkage_frobenius_2(ell, c, s):
    """max(ell / ((c^2) + ell * (s^2)), 0)"""
    return np.maximum(ell / ((c**2) + ell * (s**2)), 0)


def shrinkage_frobenius_3(ell, c, s):
    """max(1 + (ell - 1) * ((c^2) / ((ell^2) * (s^2) + c^2)), 1)"""
    return np.maximum(1 + (ell - 1) * ((c**2) / ((ell**2) * (s**2) + c**2)), 1)


def shrinkage_frobenius_4(ell, c, s):
    """(s^2 + (ell^2) * (c^2)) / ((s^2) + (ell * (c^2)))"""
    return (s**2 + (ell**2) * (c**2)) / ((s**2) + (ell * (c**2)))


def shrinkage_frobenius_6(ell, c, s):
    """1 + ((ell - 1) * (c^2)) / (((c^2) + ell * (s^2))^2)"""
    return 1 + ((ell - 1) * (c**2)) / (((c**2) + ell * (s**2)) ** 2)


def shrinkage_operator_1(ell, c, s):
    """ell"""
    return ell


def shrinkage_operator_2(ell, c, s):
    """ell  # Debiasing to population eigenvalues"""
    return ell


def shrinkage_operator_6(ell, c, s):
    """1 + ((ell - 1) / (c^2 + ell * (s^2)))"""
    return 1 + ((ell - 1) / (c**2 + ell * (s**2)))


def shrinkage_nuclear_1(ell, c, s):
    """max(1 + (ell - 1) * (1 - 2 * (s^2)), 1)"""
    return np.maximum(1 + (ell - 1) * (1 - 2 * (s**2)), 1)


def shrinkage_nuclear_2(ell, c, s):
    """max(ell / ((2 * ell - 1) * (s^2) + c^2), 1)"""
    return np.maximum(ell / ((2 * ell - 1) * (s**2) + c**2), 1)


def shrinkage_nuclear_3(ell, c, s):
    """max(ell / (c^2 + (ell^2) * (s^2)), 1)"""
    return np.maximum(ell / (c**2 + (ell**2) * (s**2)), 1)


def shrinkage_nuclear_4(ell, c, s):
    """max(ell * (c^2) + (s^2) / ell, 1)"""
    return np.maximum(ell * (c**2) + (s**2) / ell, 1)


def shrinkage_nuclear_6(ell, c, s):
    """max((ell - ((ell - 1)^2) * (c^2) * (s^2)) / (((c^2) + ell * (s^2))^2), 1)"""
    return np.maximum(
        (ell - ((ell - 1) ** 2) * (c**2) * (s**2)) / (((c**2) + ell * (s**2)) ** 2),
        1,
    )


def shrinkage_stein(ell, c, s):
    """ell / (c^2 + ell * (s^2))"""
    return ell / (c**2 + ell * (s**2))


def shrinkage_entropy(ell, c, s):
    """ell * (c^2) + s^2"""
    return ell * (c**2) + s**2


def shrinkage_divergence(ell, c, s):
    """sqrt(((ell^2) * (c^2) + ell * (s^2)) / (c^2 + (s^2) * ell))"""
    return np.sqrt(((ell**2) * (c**2) + ell * (s**2)) / (c**2 + (s**2) * ell))


def shrinkage_frechet(ell, c, s):
    """((sqrt(ell) * (c^2) + s^2)^2)"""
    return (np.sqrt(ell) * (c**2) + s**2) ** 2


def shrinkage_affine(ell, c, s):
    """((1 + c^2) * ell + (s^2)) / (1 + (c^2) + ell * (s^2))"""
    return ((1 + c**2) * ell + (s**2)) / (1 + (c**2) + ell * (s**2))


def optshrink_impl(eigenvals, gamma, loss, sigma):
    """
    Apply optimal shrinkage to the eigenvalues.

    Parameters:
    - eigenvals : array_like, shape (n,)
        Eigenvalues to be processed.
    - gamma : float
        Parameter for shrinkage.
    - loss : str
        Type of loss function to be applied.
    - sigma : float
        Standard deviation.

    Returns:
    - array_like, shape (n,)
        Processed eigenvalues.

    Apply optimal shrinkage to the given eigenvalues using the specified loss function and parameters.
    """

    sigma2 = sigma**2
    lam_plus = (1 + np.sqrt(gamma)) ** 2
    assert sigma > 0
    assert np.prod(np.shape(sigma)) == 1
    eigenvals = eigenvals / sigma2
    I = eigenvals > lam_plus
    eigenvals[~I] = 1
    str1 = f"{loss}_func = lambda lam: np.maximum(1, impl_{loss}(ell(lam), c(lam), s(lam)))"
    str2 = f"eigenvals[I] = {loss}_func(eigenvals[I])"
    exec(str1)
    exec(str2)
    eigenvals = sigma2 * eigenvals
    return eigenvals


def MarcenkoPasturIntegral(x, gamma):
    """
    Compute the integral of the Marcenko-Pastur distribution.

    Parameters:
    - x : float
        Upper limit of integration.
    - gamma : float
        Parameter of the distribution.

    Returns:
    - float
        The computed integral.

    Compute the integral of the Marcenko-Pastur distribution up to the given upper limit.
    """
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma beyond")
    lobnd = (1 - np.sqrt(gamma)) ** 2
    hibnd = (1 + np.sqrt(gamma)) ** 2
    if (x < lobnd) or (x > hibnd):
        raise ValueError("x beyond")

    dens = lambda t: np.sqrt((hibnd - t) * (t - lobnd)) / (2 * np.pi * gamma * t)
    I, _ = quad(dens, lobnd, x)
    return I


def MedianMarcenkoPastur(gamma):
    """
    Compute the median of the Marcenko-Pastur distribution.

    Parameters:
    - gamma : float
        Parameter of the distribution.

    Returns:
    - float
        The computed median.

    Iteratively compute the median of the Marcenko-Pastur distribution using a binary search algorithm.
    """

    def MarPas(x):
        return 1 - incMarPas(x, gamma, 0)

    lobnd = (1 - np.sqrt(gamma)) ** 2
    hibnd = (1 + np.sqrt(gamma)) ** 2
    change = True
    while change and (hibnd - lobnd > 0.001):
        change = False
        x = np.linspace(lobnd, hibnd, 5)
        y = np.array([MarPas(xi) for xi in x])
        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = True
        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = True

    med = (hibnd + lobnd) / 2
    return med


def incMarPas(x0, gamma, alpha):
    """
    Compute the integral of the Marcenko-Pastur distribution.

    Parameters:
    x0 : float
        Lower limit of integration.
    gamma : float
        Parameter of the distribution.
    alpha : float
        Exponent of the power function.

    Returns:
    float
        The computed integral.

    Compute the integral of the Marcenko-Pastur distribution using numerical integration.
    """
    if gamma > 1:
        raise ValueError("gammaBeyond")

    topSpec = (1 + np.sqrt(gamma)) ** 2
    botSpec = (1 - np.sqrt(gamma)) ** 2

    def IfElse(Q, point, counterPoint):
        """
        Choose between two values based on a condition.

        Parameters:
        - Q : array_like
            Boolean condition array.
        - point : array_like
            Values to choose when the condition is True.
        - counterPoint : array_like or float
            Values to choose when the condition is False.

        Returns:
        - array_like
            Selected values based on the condition.
        """
        y = point.copy()
        if np.any(~Q):
            if len(counterPoint) == 1:
                counterPoint = np.ones_like(Q) * counterPoint
            y[~Q] = counterPoint[~Q]
        return y

    def MarPas(x):
        """
        Compute the Marcenko-Pastur distribution.

        Parameters:
        - x : array_like
            Input values.

        Returns:
        - array_like
            Marcenko-Pastur distribution values corresponding to input values.
        """
        return IfElse(
            (topSpec - x) * (x - botSpec) > 0,
            np.sqrt((topSpec - x) * (x - botSpec)) / (gamma * x) / (2 * np.pi),
            np.array([0]),
        )

    if alpha != 0:
        fun = lambda x: x**alpha * MarPas(x)
    else:
        fun = MarPas

    # Numerical integration using the trapezoidal rule
    x_values = np.linspace(x0, topSpec, 1000)
    integral = np.trapz(fun(x_values), x_values)
    return integral
