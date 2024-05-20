"""
Copyright (c) 2009-2016 Iain M. Johnstone, Zongming Ma, Patrick O. Perry, and Morteza Shahram

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the authors nor the names of the contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from scipy.stats import norm, uniform, rv_continuous  # type: ignore
from scipy.optimize import root_scalar  # type: ignore
from scipy.integrate import quad  # type: ignore
import warnings
from bystro.random_matrix_theory.tracy_widom import TracyWidom
from typing import Optional, Tuple


class MarchenkoPastur(rv_continuous):
    def __init__(
        self, ndf: int, pdim: int, var: float = 1, svr: Optional[float] = None
    ):
        """
        Initialize the Marchenko-Pastur distribution.

        Args:
            ndf (int): Degrees of freedom.
            pdim (int): Dimensionality of the matrix.
            var (float, optional): Variance. Defaults to 1.
            svr (float, optional): Shape variance ratio. Defaults to None.
        """
        super().__init__()
        self.ndf = ndf
        self.pdim = pdim
        self.var = var
        self.svr = svr if svr is not None else ndf / pdim
        self.params = marchenko_pastur_par(ndf, pdim, var, self.svr)

    def _pdf(self, x: float) -> float:
        """
        Probability density function of the Marchenko-Pastur distribution.

        Args:
            x (float): Value at which to calculate the density.

        Returns:
            float: Density of the Marchenko-Pastur distribution.
        """
        return dmp(x, self.ndf, self.pdim, self.var, self.svr)

    def _cdf(self, x: float) -> float:
        """
        Cumulative distribution function of the Marchenko-Pastur distribution.

        Args:
            x (float): Value at which to calculate the CDF.

        Returns:
            float: CDF of the Marchenko-Pastur distribution.
        """
        return pmp(x, self.ndf, self.pdim, self.var, self.svr)

    def _ppf(self, q: float) -> float:
        """
        Percent point function (inverse of CDF) of the Marchenko-Pastur
        distribution.

        Args:
            q (float): Probability.

        Returns:
            float: Quantile of the Marchenko-Pastur distribution.
        """
        return qmp(q, self.ndf, self.pdim, self.var, self.svr)


class WishartMax(rv_continuous):
    def __init__(self, ndf: int, pdim: int, var: float = 1, beta: int = 1):
        """ """
        super().__init__()
        self.ndf = ndf
        self.pdim = pdim
        self.var = var
        self.beta = beta
        self.center, self.scale = wishart_max_par(ndf, pdim, var, beta)

    def _pdf(self, x: float) -> float:
        """
        Probability density function of the ?? distribution.

        Args:
            x (float): Value at which to calculate the density.

        Returns:
            float: Density of the ?? distribution.
        """
        x_transformed = (x - self.center) / self.scale
        density = dtw(x_transformed, beta=self.beta)
        out = density / self.scale
        return out

    def _cdf(self, x: float) -> float:
        """
        Cumulative distribution function of the ?? distribution.

        Args:
            x (float): Value at which to calculate the CDF.

        Returns:
            float: CDF of the ?? distribution.
        """
        x_tw = (x - self.center) / self.scale
        p = ptw(x_tw, self.beta, lower_tail=True)
        return p

    def _ppf(self, q: float) -> float:
        """
        Percent point function (inverse of CDF) of the ??
        distribution.

        Args:
            q (float): Probability.

        Returns:
            float: Quantile of the ?? distribution.
        """
        q_tw = qtw(q, beta=self.beta, lower_tail=True)
        q = self.center + q_tw * self.scale
        return q


class WishartSpike(rv_continuous):
    def __init__(
        self, spike: float, ndf: int, pdim: int, var: float = 1, beta: int = 1
    ):
        """ """
        super().__init__()
        self.spike = spike
        self.ndf = ndf
        self.pdim = pdim
        self.var = var
        self.beta = beta
        self.center, self.scale = wishart_spike_par(spike, ndf, pdim, var, beta)

    def _pdf(self, x: float) -> float:
        """
        Probability density function of the ?? distribution.

        Args:
            x (float): Value at which to calculate the density.

        Returns:
            float: Density of the ?? distribution.
        """
        d = norm.pdf(x, loc=self.center, scale=self.scale)
        return d

    def _cdf(self, x: float) -> float:
        """
        Cumulative distribution function of the ?? distribution.

        Args:
            x (float): Value at which to calculate the CDF.

        Returns:
            float: CDF of the ?? distribution.
        """
        p = norm.cdf(x, loc=self.center, scale=self.scale)
        return p

    def _ppf(self, q: float) -> float:
        """
        Percent point function (inverse of CDF) of the ??
        distribution.

        Args:
            q (float): Probability.

        Returns:
            float: Quantile of the ?? distribution.
        """
        q = norm.ppf(q, loc=self.center, scale=self.scale)
        return q


def wishart_max_par(
    ndf: int, pdim: int, var: float = 1, beta: int = 1
) -> Tuple[float, float]:
    """
    Calculate the parameters for the Wishart distribution's maximum
    eigenvalue.

    Args:
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.

    Returns:
        Tuple[float, float]: Center and scale of the distribution.
    """

    def mu(n, p):
        return (np.sqrt(n) + np.sqrt(p)) ** 2

    def sigma(n, p):
        return (np.sqrt(n) + np.sqrt(p)) * (
            (1 / np.sqrt(n) + 1 / np.sqrt(p)) ** (1 / 3)
        )

    n = ndf
    p = pdim

    if beta == 1:
        m = mu(n - 0.5, p - 0.5)
        s = sigma(n - 0.5, p - 0.5)
    elif beta == 2:
        m = mu(n - 0.5, p + 0.5)
        s = sigma(n - 0.5, p + 0.5)
    else:
        raise ValueError("`beta` must be 1 or 2")

    center = var * (m / n)
    scale = var * (s / n)
    return center, scale


def d_wishart_max(
    x: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    beta: int = 1,
    log: bool = False,
) -> float:
    """
    Calculate the density of the maximum eigenvalue of the Wishart
    distribution.

    Args:
        x (float): Value at which to calculate the density.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.
        log (bool, optional): If True, return the log density. Defaults
        to False.

    Returns:
        float: Density or log density of the maximum eigenvalue.
    """
    center, scale = wishart_max_par(ndf, pdim, var, beta)
    x_transformed = (x - center) / scale
    density = dtw(x_transformed, beta=beta)
    out = density / scale
    if log:
        return np.log(out)
    return out


def p_wishart_max(
    q: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    beta: int = 1,
    lower_tail: bool = True,
    log_p: bool = False,
) -> float:
    """
    Calculate the cumulative distribution function of the maximum eigenvalue of the Wishart distribution.

    Args:
        q (float): Quantile.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.
        lower_tail (bool, optional): If True, return P(X <= q). Defaults to True.
        log_p (bool, optional): If True, return the log CDF. Defaults to False.

    Returns:
        float: CDF or log CDF of the maximum eigenvalue.
    """
    center, scale = wishart_max_par(ndf, pdim, var, beta)
    q_tw = (q - center) / scale
    p = ptw(q_tw, beta, lower_tail)
    if log_p:
        return np.log(p)

    return p


def q_wishart_max(
    p: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    beta: int = 1,
    lower_tail: bool = True,
    log_p: bool = False,
) -> float:
    """
    Calculate the quantile function of the maximum eigenvalue of the Wishart distribution.

    Args:
        p (float): Probability.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.
        lower_tail (bool, optional): If True, return the lower tail quantile. Defaults to True.
        log_p (bool, optional): If True, p is given as a log probability. Defaults to False.

    Returns:
        float: Quantile of the maximum eigenvalue.
    """
    center, scale = wishart_max_par(ndf, pdim, var, beta)
    q_tw = qtw(p, beta=beta, lower_tail=lower_tail, log_p=log_p)
    q = center + q_tw * scale
    return q


def wishart_spike_par(
    spike: float, ndf: int, pdim: int, var: float = 1, beta: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the parameters for the spike of the Wishart distribution.

    Args:
        spike (float): Spike value.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Center and scale for the spike.
    """
    ratio = pdim / ndf
    above = spike > np.sqrt(ratio) * var
    center = np.where(
        above, (spike + var) * (1 + ratio * (var / spike)), np.nan
    )
    scale = np.where(
        above,
        (
            (spike + var)
            * np.sqrt((2 / beta) * (1 - (ratio * (var / spike) ** 2)))
            / np.sqrt(ndf)
        ),
        np.nan,
    )
    return center, scale


def d_wishart_spike(
    x: float,
    spike: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    beta: int = 1,
    log: bool = False,
) -> float:
    """
    Calculate the density of the spike of the Wishart distribution.

    Args:
        x (float): Value at which to calculate the density.
        spike (float): Spike value.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.
        log (bool, optional): If True, return the log density. Defaults to False.

    Returns:
        float: Density or log density of the spike.
    """
    centering, scaling = wishart_spike_par(spike, ndf, pdim, var, beta)
    d = norm.pdf(x, loc=centering, scale=scaling)
    return np.log(d) if log else d


def p_wishart_spike(
    q: float,
    spike: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    beta: int = 1,
    lower_tail: bool = True,
    log_p: bool = False,
) -> float:
    """
    Calculate the cumulative distribution function of the spike of
    the Wishart distribution.

    Args:
        q (float): Quantile.
        spike (float): Spike value.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.
        lower_tail (bool, optional): If True, return P(X <= q). Defaults to True.
        log_p (bool, optional): If True, return the log CDF. Defaults to False.

    Returns:
        float: CDF or log CDF of the spike.
    """
    centering, scaling = wishart_spike_par(spike, ndf, pdim, var, beta)
    p = norm.cdf(q, loc=centering, scale=scaling)
    if not lower_tail:
        p = 1 - p
    return np.log(p) if log_p else p


def q_wishart_spike(
    p: float,
    spike: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    beta: int = 1,
    lower_tail: bool = True,
    log_p: bool = False,
) -> float:
    """
    Calculate the quantile function of the spike of the Wishart distribution.

    Args:
        p (float): Probability.
        spike (float): Spike value.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        beta (int, optional): Beta parameter (1 or 2). Defaults to 1.
        lower_tail (bool, optional): If True, return the lower tail quantile. Defaults to True.
        log_p (bool, optional): If True, p is given as a log probability. Defaults to False.

    Returns:
        float: Quantile of the spike.
    """
    centering, scaling = wishart_spike_par(spike, ndf, pdim, var, beta)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    q = norm.ppf(p, loc=centering, scale=scaling)
    return q


def r_wishart_spike(n, spike, ndf=None, pdim=None, var=1, beta=1):
    center,scale = wishart_spike_par(spike, ndf, pdim, var, beta)
    x = norm.rvs(loc=center, scale=scale, size=n)
    return x


def marchenko_pastur_par(
    ndf: int, pdim: int, var: float = 1, svr: Optional[float] = None
) -> dict:
    """
    Calculate the parameters for the Marchenko-Pastur distribution.

    Args:
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        svr (float, optional): Shape variance ratio. Defaults to None.

    Returns:
        dict: Lower and upper bounds of the distribution.
    """
    if svr is None:
        svr = ndf / pdim
    inv_gamma_sqrt = np.sqrt(1 / svr)
    a = var * (1 - inv_gamma_sqrt) ** 2
    b = var * (1 + inv_gamma_sqrt) ** 2
    return {"lower": a, "upper": b}


def dmp(
    x: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    svr: Optional[float] = None,
    log: bool = False,
) -> float:
    """
    Calculate the density of the Marchenko-Pastur distribution.

    Args:
        x (float): Value at which to calculate the density.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        svr (float, optional): Shape variance ratio. Defaults to None.
        log (bool, optional): If True, return the log density. Defaults to False.

    Returns:
        float: Density or log density of the Marchenko-Pastur distribution.
    """
    if svr is None:
        svr = ndf / pdim
    params = marchenko_pastur_par(ndf, pdim, var, svr)
    a, b = params["lower"], params["upper"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inside = (x > a) & (x < b)
        sqrt_term = np.sqrt((x - a) * (b - x))
        density = svr / (2 * np.pi * var * x) * sqrt_term
        density = np.where(inside, density, 0)
        if log:
            density = np.log(
                density, out=np.full_like(density, -np.inf), where=density > 0
            )
    return density


def pmp(
    q: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    svr: Optional[float] = None,
    lower_tail: bool = True,
    log_p: bool = False,
) -> float:
    """
    Calculate the cumulative distribution function of the Marchenko-Pastur distribution.

    Args:
        q (float): Quantile.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        svr (float, optional): Shape variance ratio. Defaults to None.
        lower_tail (bool, optional): If True, return P(X <= q). Defaults to True.
        log_p (bool, optional): If True, return the log CDF. Defaults to False.

    Returns:
        float: CDF or log CDF of the Marchenko-Pastur distribution.
    """
    if svr is None:
        svr = ndf / pdim
    params = marchenko_pastur_par(ndf, pdim, var, svr)
    a, b = params["lower"], params["upper"]

    def integrand(x):
        return dmp(x, ndf, pdim, var, svr)

    if lower_tail:
        p, _ = quad(integrand, a, q)
        p += (1 - svr) if (svr < 1 and q >= 0) else 0
    else:
        p, _ = quad(integrand, q, b)
        p += (1 - svr) if (svr < 1 and q <= 0) else 0

    if log_p:
        p = np.log(p)
    return p


def qmp(
    p: float,
    ndf: int,
    pdim: int,
    var: float = 1,
    svr: Optional[float] = None,
    lower_tail: bool = True,
    log_p: bool = False,
) -> float:
    """
    Calculate the quantile function of the Marchenko-Pastur distribution.

    Args:
        p (float): Probability.
        ndf (int): Degrees of freedom.
        pdim (int): Dimensionality of the matrix.
        var (float, optional): Variance. Defaults to 1.
        svr (float, optional): Shape variance ratio. Defaults to None.
        lower_tail (bool, optional): If True, return the lower tail quantile. Defaults to True.
        log_p (bool, optional): If True, p is given as a log probability. Defaults to False.

    Returns:
        float: Quantile of the Marchenko-Pastur distribution.
    """
    if svr is None:
        svr = ndf / pdim
    params = marchenko_pastur_par(ndf, pdim, var, svr)

    if log_p:
        p = np.exp(p)

    if not lower_tail:
        p = 1 - p

    def func(x):
        return pmp(x, ndf, pdim, var, svr) - p

    result = root_scalar(func, bracket=[params["lower"], params["upper"]])
    return result.root


def rmp(n, ndf=None, pdim=None, var=1, svr=None):
    u = uniform.rvs(size=n)
    return np.array([qmp(ui, ndf, pdim, var, svr) for ui in u])


def dtw(x, beta=1, log_p=False):
    if beta not in [1, 2, 4]:
        raise ValueError("'beta' must be '1', '2', or '4'.")
    mod = TracyWidom(beta=beta)
    if log_p:
        return np.log(mod.pdf(x))
    return mod.pdf(x)


def ptw(q, beta=1, lower_tail=True, log_p=False):
    if beta not in [1, 2, 4]:
        raise ValueError("'beta' must be '1', '2', or '4'.")
    mod = TracyWidom(beta=beta)

    p = mod.cdf(q) if lower_tail else 1 - mod.cdf(q)

    if log_p:
        return np.log(p)
    return p


def qtw(p, beta=1, lower_tail=True, log_p=False):
    if beta not in [1, 2, 4]:
        raise ValueError("'beta' must be '1', '2', or '4'.")
    mod = TracyWidom(beta=beta)
    if log_p:
        p = np.exp(p)
    if lower_tail:
        return mod.cdfinv(p)

    return mod.cdfinv(1 - p)


def rtw(n, beta=1):
    rng = np.random.default_rng()
    u = rng.uniform(low=0.0, high=1.0, size=n)
    mod = TracyWidom(beta=beta)
    return mod.cdfinv(u)
