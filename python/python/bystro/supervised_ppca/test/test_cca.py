import numpy as np
import numpy.linalg as la
from ..cannonical_correlation_analysis import CCA


def generate_data(n_groups=4):
    rng = np.random.default_rng(2021)
    N = 1000000  # Samples per group
    eps = 0.3
    p_list = rng.integers(10, high=60, size=n_groups)
    Ls = 1  # Shared components
    Li = 1  # Individual components
    p = int(np.sum(p_list))
    B_true_list = [
        0.2 * rng.normal(size=(Li, p_list[i])) for i in range(n_groups)
    ]
    W_true = rng.normal(size=(Ls, int(np.sum(p_list))))
    cov = eps * np.eye(p)
    cov = cov + np.dot(W_true.T, W_true)
    p_list2 = np.zeros(n_groups + 1)
    p_list2[1:] = p_list
    for i in range(n_groups):
        st = int(np.sum(p_list2[: i + 1]))
        nd = int(np.sum(p_list2[: (i + 2)]))
        a = cov[st:nd, st:nd]
        bbx = np.dot(B_true_list[i].T, B_true_list[i])
        cov[st:nd, st:nd] = a + bbx

    L = la.cholesky(cov)
    X_o = rng.normal(size=(N, p))
    X = np.dot(X_o, L)
    idxs = np.zeros(p)
    count = 0
    for i in range(n_groups):
        idxs[count : count + p_list[i]] = i
        count += p_list[i]

    return X, W_true, B_true_list, cov, idxs, eps


def test_methods():
    X, W_true, B_true_list, cov, idxs, eps = generate_data()

    model = CCA(1)
    model.fit(X, groups=idxs)

    assert np.abs(model.sigma2_ - eps ** 2) < 0.3
