import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from scipy.stats import ortho_group  # type: ignore
from scipy.stats import binom  # type: ignore

from bystro.supervised_ppca.gf_dropout_pt import PPCADropout


def PPCA_generate_data(N=10000, L=5, p=30, phi=1.0, sigma=1.0):
    rng = np.random.default_rng(2021)
    S = rng.normal(size=(N, L))
    W_o = ortho_group.rvs(p) * np.sqrt(p)
    W = W_o[:L] * 3
    W[0] = 0.3 * W[0]

    X_hat = np.dot(S, W)
    X_noise = sigma * rng.normal(size=(N, p))
    X = X_hat + X_noise

    logits = phi * S[:, 0] * 5
    probs = np.exp(logits) / (1 + np.exp(logits))
    y = binom.rvs(1, probs, random_state=42)
    return X, y, X_hat, S, W, logits


def test_ppca():
    X, y, X_hat, S, W, logits = PPCA_generate_data(L=3, p=200)
    training_options = {"n_iterations": 20}
    model = PPCADropout(
        2, mu=100.0, gamma=10.0, delta=5.0, training_options=training_options
    )
    model.fit(X, y)
    S_ = model.transform(X)
    Y_hat = S_[:, 0] + model.B_
    model2 = LogisticRegression(C=0.1)
    model2.fit(X, y)
    y_hat = model2.decision_function(X)
    roc_model = roc_auc_score(y, y_hat)
    roc_linear = roc_auc_score(y, Y_hat)
    assert roc_model > roc_linear - 0.05

    training_options = {"n_iterations": 20, "use_gpu": False}
    model = PPCADropout(
        2, mu=100.0, gamma=10.0, delta=5.0, training_options=training_options
    )
    model.fit(X, y)

    training_options = {"n_iterations": 20, "use_gpu": False}
    model = PPCADropout(
        2, mu=100.0, gamma=10.0, delta=5.0, training_options=training_options
    )
    model.fit(X, logits, task="regression")
