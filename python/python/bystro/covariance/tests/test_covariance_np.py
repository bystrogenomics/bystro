import numpy as np
import numpy.linalg as la
from bystro.covariance._covariance_np import (
    EmpiricalCovariance,
    BayesianCovariance,
    LinearShrinkageCovariance,
    NonLinearShrinkageCovariance,
    NonnegativeCovariance,
)


def test_empirical_covariance():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100000, 10))

    model = EmpiricalCovariance()
    model.fit(X)

    assert model.covariance is not None
    assert model.covariance.shape == (10, 10)

    s_vals = la.svd(model.covariance, compute_uv=False)
    assert np.abs(1 - s_vals[0]) <= 5e-2


def test_bayesian_covariance():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100000, 10))

    model = BayesianCovariance()
    model.fit(X)

    assert model.covariance is not None
    assert model.covariance.shape == (10, 10)

    s_vals = la.svd(model.covariance, compute_uv=False)
    assert np.abs(1 - s_vals[0]) <= 5e-2


def test_linearshrinkage_covariance():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100000, 10))

    model = LinearShrinkageCovariance()
    model.fit(X)

    assert model.covariance is not None
    assert model.covariance.shape == (10, 10)

    s_vals = la.svd(model.covariance, compute_uv=False)
    assert np.abs(1 - s_vals[0]) <= 5e-2


def test_nonlinear_covariance():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100000, 10))

    model = NonLinearShrinkageCovariance()
    model.fit(X)

    assert model.covariance is not None
    assert model.covariance.shape == (10, 10)

    s_vals = la.svd(model.covariance, compute_uv=False)
    assert np.abs(1 - s_vals[0]) <= 5e-2


def test_nonnegative_covariance():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100000, 10))

    model = NonnegativeCovariance()
    model.fit(X)

    assert model.covariance is not None
    assert model.covariance.shape == (10, 10)

    assert np.all(model.covariance >= 0)
    eigvals = np.linalg.eigvals(model.covariance)
    assert np.all(eigvals >= 0)

    s_vals = la.svd(model.covariance, compute_uv=False)
    assert np.abs(1 - s_vals[0]) <= 5e-2
