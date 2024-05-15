import numpy as np
from scipy.stats import uniform # type: ignore 
from scipy.stats import norm # type: ignore 
from scipy.optimize import root_scalar # type: ignore 
from scipy.integrate import quad # type: ignore 
import warnings
from bystro.random_matrix_theory.tracy_widom import TracyWidom


def wishart_max_par(ndf, pdim, var=1, beta=1):
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


def d_wishart_max(x, ndf, pdim, var=1, beta=1, log=False):
    center, scale = wishart_max_par(ndf, pdim, var, beta)
    x_transformed = (x - center) / scale
    density = dtw(x_transformed, beta=beta)
    out = density / scale
    if log:
        return np.log(out)
    return out


def p_wishart_max(q, ndf, pdim, var=1, beta=1, lower_tail=True, log_p=False):
    center, scale = wishart_max_par(ndf, pdim, var, beta)
    q_tw = (q - center) / scale
    p = ptw(q_tw, beta, lower_tail)
    if log_p:
        return np.log(p)

    return p


def q_wishart_max(p, ndf, pdim, var=1, beta=1, lower_tail=True, log_p=False):
    center, scale = wishart_max_par(ndf, pdim, var, beta)
    q_tw = qtw(p, beta=beta, lower_tail=lower_tail, log_p=log_p)
    q = center + q_tw * scale
    return q


def wishart_spike_par(spike, ndf=None, pdim=None, var=1, beta=1):
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
    return {"centering": center, "scaling": scale}


def d_wishart_spike(x, spike, ndf=None, pdim=None, var=1, beta=1, log=False):
    params = wishart_spike_par(spike, ndf, pdim, var, beta)
    d = norm.pdf(x, loc=params["centering"], scale=params["scaling"])
    return np.log(d) if log else d


def p_wishart_spike(
    q, spike, ndf=None, pdim=None, var=1, beta=1, lower_tail=True, log_p=False
):
    params = wishart_spike_par(spike, ndf, pdim, var, beta)
    p = norm.cdf(q, loc=params["centering"], scale=params["scaling"])
    if not lower_tail:
        p = 1 - p
    return np.log(p) if log_p else p


def q_wishart_spike(
    p, spike, ndf=None, pdim=None, var=1, beta=1, lower_tail=True, log_p=False
):
    params = wishart_spike_par(spike, ndf, pdim, var, beta)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    q = norm.ppf(p, loc=params["centering"], scale=params["scaling"])
    return q


def r_wishart_spike(n, spike, ndf=None, pdim=None, var=1, beta=1):
    params = wishart_spike_par(spike, ndf, pdim, var, beta)
    x = norm.rvs(loc=params["centering"], scale=params["scaling"], size=n)
    return x


def marchenko_pastur_par(ndf=None, pdim=None, var=1, svr=None):
    if svr is None:
        svr = ndf / pdim
    inv_gamma_sqrt = np.sqrt(1 / svr)
    a = var * (1 - inv_gamma_sqrt) ** 2
    b = var * (1 + inv_gamma_sqrt) ** 2
    return {"lower": a, "upper": b}


def dmp(x, ndf=None, pdim=None, var=1, svr=None, log=False):
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


def pmp(q, ndf=None, pdim=None, var=1, svr=None, lower_tail=True, log_p=False):
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


def qmp(p, ndf=None, pdim=None, var=1, svr=None, lower_tail=True, log_p=False):
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
    if beta not in [1, 2, 4 ]:
        raise ValueError("'beta' must be '1', '2', or '4'.")
    mod = TracyWidom(beta=beta)
    if log_p:
        return np.log(mod.pdf(x))
    return mod.pdf(x)


def ptw(q, beta=1, lower_tail=True, log_p=False):
    if beta not in [1, 2, 4 ]:
        raise ValueError("'beta' must be '1', '2', or '4'.")
    mod = TracyWidom(beta=beta)

    p = mod.cdf(q) if lower_tail else 1 - mod.cdf(q)

    if log_p:
        return np.log(p)
    return p


def qtw(p, beta=1, lower_tail=True, log_p=False):
    if beta not in [1, 2, 4 ]:
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
