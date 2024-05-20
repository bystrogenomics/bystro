import numpy as np
from numpy.testing import assert_array_almost_equal
from bystro.covariance.geodesics import (
    geodesic_euclid,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic,
)


def test_geodesic_euclid():
    rng1 = np.random.default_rng(seed=2021)
    rng2 = np.random.default_rng(seed=2023)
    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0.3
    result = geodesic_euclid(A, B, alpha)
    expected = np.array(
        [
            [1.28787409, 0.67259424, 0.66738804],
            [0.67259424, 0.64066451, 0.6468242],
            [0.66738804, 0.6468242, 0.79058781],
        ]
    )
    assert_array_almost_equal(result, expected, decimal=6)

    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0
    result = geodesic_euclid(A, B, alpha)
    expected = A
    assert_array_almost_equal(result, expected, decimal=6)


def test_geodesic_logeuclid():
    rng1 = np.random.default_rng(seed=2021)
    rng2 = np.random.default_rng(seed=2023)

    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0.3
    result = geodesic_logeuclid(A, B, alpha)
    expected = np.array(
        [
            [0.25249811, 0.35078989, 0.23053012],
            [0.35078989, 0.49934116, 0.3205249],
            [0.23053012, 0.3205249, 0.24452255],
        ]
    )
    assert_array_almost_equal(result, expected, decimal=6)

    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 1
    result = geodesic_logeuclid(A, B, alpha)
    expected = B
    assert_array_almost_equal(result, expected, decimal=6)


def test_geodesic_riemann():
    rng1 = np.random.default_rng(seed=2021)
    rng2 = np.random.default_rng(seed=2023)
    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0.3
    result = geodesic_riemann(A, B, alpha)
    expected = np.array(
        [
            [0.04489211, 0.07002845, 0.02882102],
            [0.07002845, 0.16552368, 0.06916183],
            [0.02882102, 0.06916183, 0.06972414],
        ]
    )

    assert_array_almost_equal(result, expected, decimal=6)

    rng1 = np.random.default_rng(seed=2021)
    rng2 = np.random.default_rng(seed=2023)
    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0.5
    result = geodesic_riemann(A, B, alpha)
    expected = np.array(
        [
            [0.02696476, 0.0718294, 0.03588318],
            [0.0718294, 0.20845467, 0.13392089],
            [0.03588318, 0.13392089, 0.16357681],
        ]
    )

    assert_array_almost_equal(result, expected, decimal=6)


def test_geodesic():
    rng1 = np.random.default_rng(seed=2020)
    rng2 = np.random.default_rng(seed=2022)

    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0.2
    result = geodesic(A, B, alpha, "euclid")
    expected = np.array(
        [
            [1.07309276, 1.12383006, 0.96590731],
            [1.12383006, 1.3270369, 1.01724117],
            [0.96590731, 1.01724117, 0.88887129],
        ]
    )

    assert_array_almost_equal(result, expected, decimal=6)

    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0.4
    result = geodesic(A, B, alpha, "logeuclid")
    expected = np.array(
        [
            [1.24282468, 1.13790717, 1.23823539],
            [1.13790717, 1.07234696, 1.09855237],
            [1.23823539, 1.09855237, 1.32320416],
        ]
    )

    assert_array_almost_equal(result, expected, decimal=6)

    A = rng1.random((3, 3))
    A = A @ A.T
    B = rng2.random((3, 3))
    B = B @ B.T
    alpha = 0.6
    result = geodesic(A, B, alpha, "riemann")
    expected = np.array(
        [
            [1.51789576, 1.21423412, 0.37889893],
            [1.21423412, 1.1119412, 0.33642569],
            [0.37889893, 0.33642569, 0.23550756],
        ]
    )

    assert_array_almost_equal(result, expected, decimal=6)
