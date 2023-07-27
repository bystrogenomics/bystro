import numpy as np
from bystro.tadarrr._reduced_rank_pt import ReducedRankPT


def test_rrr_fit():
    rng = np.random.default_rng()

    model = ReducedRankPT()
    N_pred = 10000
    p = 10
    q = 5

    coefs = rng.normal(shape=(p, q))
    X = rng.normal(shape=(N_pred, p))
    Y_nonoise = np.dot(X, coefs)
    Y = Y_nonoise + rng.normal(shape=(N_pred, q))
    model.fit(X, Y)

    Y_hat = model.predict(X)
    mse = np.mean((Y_hat - Y_nonoise) ** 2)
    assert mse <= 0.1
