import numpy as np
import numpy.linalg as la
import scipy.stats as st  # type: ignore
from bystro.supervised_ppca.ppca_analytic_np import PPCAanalytic


def test_methods():
    rng = np.random.default_rng(2021)
    N, p = 100000, 10
    L = 2
    W = rng.normal(size=(L, p))
    sigma = 0.2
    cov = np.dot(W.T, W) + sigma * np.eye(p)
    X = st.multivariate_normal.rvs(
        mean=np.zeros(p), cov=cov, size=N, random_state=1993
    )
    model = PPCAanalytic(n_components=L)
    model.fit(X)
    cov_est = model.get_covariance()
    cov_emp = np.dot(X.T, X) / X.shape[0]
    s1 = la.norm(cov_emp - cov)
    s2 = la.norm(cov_est - cov)
    assert np.abs(s2 - s1) <= 0.01

    idx_subset = np.zeros(p)
    n_sub = 8
    idx_subset[:n_sub] = 1

    model.transform_subset(X[:, :n_sub], idx_subset == 1)
