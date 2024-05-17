import numpy as np
import numpy.linalg as la
from bystro.covariance.operator_shrinkage import operator_shrinkage


def test_operator_shrinkage():
    p = 10
    cov = 5 * np.eye(p)

    n = 1000000
    gamma = p / n
    rng = np.random.default_rng(2021)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    est_covariance = np.dot(X.T, X) / n

    s_vals = la.svd(est_covariance, compute_uv=False)
    shrunk_vals = operator_shrinkage(s_vals, gamma)
    assert np.abs(shrunk_vals[0] - s_vals[0]) <= 3e-1

    n = 100000
    gamma = p / n
    rng = np.random.default_rng(2021)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    est_covariance = np.dot(X.T, X) / n

    s_vals = la.svd(est_covariance, compute_uv=False)
    shrunk_vals = operator_shrinkage(s_vals, gamma)
    assert np.abs(shrunk_vals[0] - s_vals[0]) <= 3e-1

    n = 10000
    gamma = p / n
    rng = np.random.default_rng(2021)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    est_covariance = np.dot(X.T, X) / n

    s_vals = la.svd(est_covariance, compute_uv=False)
    shrunk_vals = operator_shrinkage(s_vals, gamma)
    assert np.abs(shrunk_vals[0] - s_vals[0]) <= 3e-1

    n = 1000
    gamma = p / n
    rng = np.random.default_rng(2021)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    est_covariance = np.dot(X.T, X) / n

    s_vals = la.svd(est_covariance, compute_uv=False)
    shrunk_vals = operator_shrinkage(s_vals, gamma)
    assert np.abs(shrunk_vals[0] - s_vals[0]) <= 3e-1

    n = 100
    gamma = p / n
    rng = np.random.default_rng(2021)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    est_covariance = np.dot(X.T, X) / n

    s_vals = la.svd(est_covariance, compute_uv=False)
    shrunk_vals = operator_shrinkage(s_vals, gamma)
    assert np.abs(shrunk_vals[0] - s_vals[0]) <= 3e-1

    n = 10
    gamma = p / n
    rng = np.random.default_rng(2021)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    est_covariance = np.dot(X.T, X) / n

    s_vals = la.svd(est_covariance, compute_uv=False)
    shrunk_vals = operator_shrinkage(s_vals, gamma)
    assert np.abs(shrunk_vals[0] - s_vals[0]) <= 3e-1

    n = 1
    gamma = p / n
    rng = np.random.default_rng(2021)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    est_covariance = np.dot(X.T, X) / n

    s_vals = la.svd(est_covariance, compute_uv=False)
    shrunk_vals = operator_shrinkage(s_vals, gamma)
    assert np.abs(shrunk_vals[0] - s_vals[0]) <= 3e-1
