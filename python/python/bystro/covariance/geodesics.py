"""
LICENSE FROM pyRiemann

https://github.com/pyRiemann/pyRiemann/blob/master/LICENSE

Copyright (c) 2015-2024, authors of pyRiemann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from scipy.linalg import sqrtm, logm, expm, inv, fractional_matrix_power  # type: ignore
import numpy as np
from typing import Union, Callable


def geodesic_euclid(
    A: np.ndarray, B: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    r"""Euclidean geodesic between matrices.

    The matrix at position :math:`\alpha` on the Euclidean geodesic
    between two matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is:

    .. math::
        \mathbf{C} = (1-\alpha) \mathbf{A} + \alpha \mathbf{B}

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, m)
        First matrices.
    B : ndarray, shape (..., n, m)
        Second matrices.
    alpha : float, default=0.5
        Position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, m)
        Matrices on the Euclidean geodesic.
    """
    return (1 - alpha) * A + alpha * B


def geodesic_logeuclid(
    A: np.ndarray, B: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    r"""Log-Euclidean geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the Log-Euclidean geodesic
    between two SPD/HPD matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` is:

    .. math::
        \mathbf{C} = \exp \left( (1-\alpha) \log(\mathbf{A})
                     + \alpha \log(\mathbf{B}) \right)

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices.
    alpha : float, default=0.5
        Position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices on the Log-Euclidean geodesic.
    """
    return expm((1 - alpha) * logm(A) + alpha * logm(B))


def geodesic_riemann(
    A: np.ndarray, B: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    r"""Affine-invariant Riemannian geodesic between SPD/HPD matrices.

    The matrix at position :math:`\alpha` on the affine-invariant Riemannian
    geodesic between two SPD/HPD matrices :math:`\mathbf{A}` and
    :math:`\mathbf{B}` is:

    .. math::
        \mathbf{C} = \mathbf{A}^{1/2} \left( \mathbf{A}^{-1/2} \mathbf{B}
                     \mathbf{A}^{-1/2} \right)^\alpha \mathbf{A}^{1/2}

    :math:`\mathbf{C}` is equal to :math:`\mathbf{A}` if :math:`\alpha` = 0,
    and :math:`\mathbf{B}` if :math:`\alpha` = 1.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices.
    alpha : float, default=0.5
        Position on the geodesic.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        SPD/HPD matrices on the affine-invariant Riemannian geodesic.
    """
    sA = sqrtm(A)
    isA = inv(sA)
    C = isA @ B @ isA
    D = fractional_matrix_power(C, alpha)
    E = sA @ D @ sA
    return E


def geodesic(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
    metric: Union[str, Callable] = "riemann",
) -> np.ndarray:
    """Geodesic between matrices according to a metric.

    Return the matrix at the position alpha on the geodesic between matrices
    A and B according to a metric.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First matrices.
    B : ndarray, shape (..., n, n)
        Second matrices.
    alpha : float
        Position on the geodesic.
    metric : string | callable, default="riemann"
        Metric used for geodesic, can be: "euclid", "logeuclid", "riemann",
        or a callable function.

    Returns
    -------
    C : ndarray, shape (..., n, n)
        Matrices on the geodesic.
    """
    geodesic_functions = {
        "euclid": geodesic_euclid,
        "logeuclid": geodesic_logeuclid,
        "riemann": geodesic_riemann,
    }
    if isinstance(metric, str):
        if metric not in geodesic_functions.keys():
            raise ValueError(
                f"Unknown function name '{metric}'. Must be one of "
                f"{' '.join(geodesic_functions.keys())}"
            )
        metric = geodesic_functions[metric]
    elif not hasattr(metric, "__call__"):
        raise ValueError(
            f"Argument must be a string or a callable, Got {type(metric)})."
        )
    C = metric(A, B, alpha)
    return C
