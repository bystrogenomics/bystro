import numpy as np
from ..mixture_tada_em import MVTadaPoissonEM, MVTadaZipEM


def test_fit():
    N = 1000
    p = 10
    K = 2

    rng = np.random.default_rng(2021)

    Z_true = rng.integers(0, high=K, size=N)
    Lambda = rng.integers(0, high=50, size=(K, p))
    Alphas = 0.2 * np.ones((K, p))

    Lambda = Lambda * 1.0
    Lambda[0, int(p / 2) :] = 0
    Lambda[1, : int(p / 2)] = 0
    for k in range(K):
        Lambda[k] = Lambda[k] / np.mean(Lambda[k]) * 10

    X = np.zeros((N, p))
    for i in range(N):
        idx = int(Z_true[i])
        X[i] = rng.poisson(Lambda[idx])
        zerod_out = rng.binomial(1, Alphas[idx])
        X[i, zerod_out == 1] = 0

    model = MVTadaZipEM(K=K)
    model.fit(X)

    model = MVTadaPoissonEM(K=K)
    model.fit(X)
