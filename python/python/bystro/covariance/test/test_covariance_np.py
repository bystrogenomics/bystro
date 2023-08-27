import numpy as np
import numpy.linalg as la
from .._covariance_np import EmpiricalCovariance, BayesianCovariance


def test_empirical_covariance():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100000, 10))

    model = EmpiricalCovariance()
    model.fit(X)

    s_vals = la.svd(model.covariance, compute_uv=False)
    assert np.abs(1 - s_vals[0]) <= 5e-2


def test_bayesian_covariance():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100000, 10))

    model = BayesianCovariance()
    model.fit(X)

    s_vals = la.svd(model.covariance, compute_uv=False)
    assert np.abs(1 - s_vals[0]) <= 5e-2
