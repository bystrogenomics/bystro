"""
TracyWidom.py
Generate Tracy-Widom distribution functions
Author: Yao-Yuan Mao (yymao)
Project website: https://github.com/yymao/TracyWidom

The MIT License (MIT)
Copyright (c) 2013-2021 Yao-Yuan Mao
http://opensource.org/licenses/MIT
"""

import numpy as np
from scipy.interpolate import interp1d  # type: ignore
from scipy.misc import derivative  # type: ignore
from scipy.optimize import brentq  # type: ignore
import pickle

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent  # folder of the script
BINARY_STRING_TW_PATH = str(SCRIPT_DIR / "binary_string_tw.p")

with open(BINARY_STRING_TW_PATH, "rb") as f:
    ddict = pickle.load(f)

_digits_1 = ddict["d1"]
_digits_2 = ddict["d2"]
_digits_4 = ddict["d4"]

_tau0 = 0.872371414954127

_para_n = [
    [_tau0**0.5 / 2.0**0.25, 1.0 / 24.0, 2.0 ** (-0.5) / 3.0, 1.0 / 16.0],
    [_tau0, 1.0 / 12.0, 0, 1.0 / 8.0],
    [
        _tau0**0.5 / 2.0 ** (38.0 / 48.0),
        1.0 / 6.0,
        -(2.0**0.5) / 3.0,
        1.0 / 16.0,
    ],
]

_para_p = [
    [0.25 / (np.pi**0.5), 0, 2.0 / 3.0, 0.75],
    [1.0 / (16.0 * np.pi), 0, 4.0 / 3.0, 1.5],
    [1.0 / (512.0 * np.pi), 0, 8.0 / 3.0, 3.0],
]


def _f(x, N, a, b, c, u=0, v=0):
    s = x**1.5
    return N * np.exp(-s * (s * a + b)) / x**c * (1.0 + (u + v / s) / s)


def _dlnf_dx(x, a, b, c, u=0, v=0):
    s = x**1.5
    return (
        -(
            3.0 * a * s * s
            + 1.5 * b * s
            + c
            + 1.5 * (s * u + v + v) / ((s + u) * s + v)
        )
        / x
    )


def _find_u_v(x, f, dlnf, N, a, b, c):
    # log f = log N - a x^3 - b x^1.5 - c log x + log1p ( (u+v/s)/s )
    s = x**1.5
    xd2 = x * 2.0 * (dlnf - _dlnf_dx(x, a, b, c))

    def vv(uu):
        return -s * ((xd2 + 3.0) * uu + xd2 * s) / (xd2 + 6.0)

    def fu(uu):
        return f / _f(x, N, a, b, c, uu, vv(uu)) - 1.0

    for k1, k2 in [(-1, 1), (-5, -3), (715, 725)]:
        try:
            u = brentq(fu, k1, k2, disp=False)
        except ValueError:
            continue
        else:
            break
    else:
        raise ValueError("Cannot find u with brentq")
    v = vv(u)
    return u, v


def _finv(y, N, a, b, c, u=0, v=0):
    # a * x**3 + b*x**1.5 + c * log x + log y - log N
    ya = np.asanyarray(y)
    xa = np.ones_like(ya)
    flag = ya > 0
    logyn = np.log(ya[flag] / N)
    xa[~flag] = np.inf
    for i in range(3):
        ca = (
            np.log(xa[flag]) * c
            + logyn
            - np.log1p(u / xa[flag] ** 1.5 + v / xa[flag] ** 3)
        )
        xa[flag] = (
            (np.sqrt(b * b - 4.0 * a * ca) - b) / (a * 2.0) if a else ca / (-b)
        ) ** (2.0 / 3.0)
    return xa


class TracyWidom(object):
    """
    Provide the Tracy-Widom distribution functions for beta = 1, 2, or 4.
    We use the tables in http://www.cl.cam.ac.uk/~aib29/TWinSplus.pdf
    and the asymptotics in http://arxiv.org/abs/1111.2761
    """

    def __init__(self, beta=2):
        """
        Construnct a TracyWidom class for a given beta.

        Parameters
        ----------
        beta : int, optional
            The beta value of the Tracy-Widom distribution.
            Can only be 1, 2, or 4; otherwise a ValueError will raise.
            Default value is 2.

        Returns
        -------
        A TracyWidom class instance with member functions cdf, pdf, and cdfinv.
        """
        b = int(beta)
        if b == 1:
            digits = _digits_1
            xlim = (-389, 360)
        elif b == 2:
            digits = _digits_2
            xlim = (-389, 250)
        elif b == 4:
            digits = _digits_4
            xlim = (-399, 70)
        else:
            raise ValueError("beta needs to be 1, 2, or 4.")

        self.beta = b
        ib = [1, 2, 4].index(b)

        x = np.arange(*xlim, dtype=np.int32).astype(float) * 1.0e-2
        y = np.frombuffer(digits, dtype=np.int32).astype(float) * 1.0e-6
        self.__xlim = (x[1], x[-2])
        self.__cdf = interp1d(x, y, kind="cubic", bounds_error=False)

        dlnf = derivative(
            lambda xx: np.log(self.__cdf(-xx)), -x[1], dx=3.33e-3, order=7
        )
        self.__para_n = _para_n[ib] + list(
            _find_u_v(-x[1], y[1], dlnf, *_para_n[ib])
        )
        self.__asym_n = lambda xx: _f(-xx, *self.__para_n)
        self.__asym_inv_n = lambda yy: -(_finv(yy, *self.__para_n))

        dlnf = derivative(
            lambda xx: np.log(1.0 - self.__cdf(xx)), x[-2], dx=3.33e-3, order=7
        )
        self.__para_p = _para_p[ib] + list(
            _find_u_v(x[-2], 1.0 - y[-2], dlnf, *_para_p[ib])
        )
        self.__asym_p = lambda xx: 1.0 - _f(xx, *self.__para_p)
        self.__asym_inv_p = lambda yy: _finv(1.0 - yy, *self.__para_p)

        x = np.linspace(-8, 4, 2401)
        y = self.cdf(x)
        self.__ylim = (y[0], y[-1])
        self.__cdfinv = interp1d(self.cdf(x), x, bounds_error=False)

    def cdf(self, x):
        """
        Return the cumulative distribution function at x.
        cdf(x) = P(TW < x)

        Parameters
        ----------
        x : float or array-like

        Returns
        -------
        y : float or array-like
            y = cdf(x)
        """
        xa = np.asanyarray(x)
        scalar = xa.ndim == 0
        if scalar:
            xa = xa.flatten()
        y = self.__cdf(xa)
        flag = xa < self.__xlim[0]
        y[flag] = self.__asym_n(xa[flag])
        flag = xa > self.__xlim[1]
        y[flag] = self.__asym_p(xa[flag])
        return y[0] if scalar else y

    def pdf(self, x):
        """
        Return the probability distribution function at x.
        pdf(x) = d P(TW < x) / dx.

        Parameters
        ----------
        x : float or array-like

        Returns
        -------
        y : float or array-like
            y = pdf(x)
        """
        xa = np.asanyarray(x)
        return derivative(self.cdf, xa, dx=0.08, order=5)

    def cdfinv(self, x):
        """
        Return the inverse cumulative distribution function at x.
        cdfinv(x) = cdf^{-1}(x)

        Parameters
        ----------
        x : float or array-like
            Only values in (0, 1) are valid.

        Returns
        -------
        y : float or array-like
            y = cdfinv(x)
        """
        xa = np.asanyarray(x)
        scalar = xa.ndim == 0
        if scalar:
            xa = xa.flatten()
        y = self.__cdfinv(xa)
        flag = xa < self.__ylim[0]
        y[flag] = self.__asym_inv_n(xa[flag])
        flag = xa > self.__ylim[1]
        y[flag] = self.__asym_inv_p(xa[flag])
        return y[0] if scalar else y
