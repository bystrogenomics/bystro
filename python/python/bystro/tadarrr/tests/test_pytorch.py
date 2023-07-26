import numpy as np
import numpy.random as rand
from bystro.tadarrr._reduced_rank_pt import ReducedRankPT


def test_rrr_fit():
    rand.seed(1993)

    model = ReducedRankPT()
    N_pred = 10000
    p = 10
    q = 5

    coefs = rand.randn(p, q)
    X = rand.randn(N_pred, p)
    Y_nonoise = np.dot(X, coefs)
    Y = Y_nonoise + rand.randn(N_pred, q)
    model.fit(X, Y)

    Y_hat = model.predict(X)
    mse = np.mean((Y_hat - Y_nonoise) ** 2)
    assert mse <= 0.1
