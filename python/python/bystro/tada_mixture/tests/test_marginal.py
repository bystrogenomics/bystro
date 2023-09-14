import numpy as np
from ..mixture_tada_marginal import MVTadaPoissonML


def test_fit():
    N = 1000
    p = 10
    K = 2

    rng = np.random.default_rng(2021)

    Z_true = rng.integers(0, high=K, size=N)
    Lambda = rng.integers(0, high=50, size=(K, p))

    Lambda = Lambda * 1.0
    Lambda[0, int(p / 2) :] = 0
    Lambda[1, : int(p / 2)] = 0
    for k in range(K):
        Lambda[k] = Lambda[k] / np.mean(Lambda[k]) * 10

    X = np.zeros((N, p))
    for i in range(N):
        idx = int(Z_true[i])
        X[i] = rng.poisson(Lambda[idx])

    model = MVTadaPoissonML(K=K, training_options={"n_iterations": 2})
    model.fit(X)
