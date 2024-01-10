import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from scipy.stats import ortho_group  # type: ignore
from scipy.stats import binom  # type: ignore

from bystro.supervised_ppca.gf_adversarial_pt import PPCAAdversarial


def PPCA_generate_data(N=10000, L=5, p=30, phi=1.0, sigma=1.0):
    rng = np.random.default_rng(2021)
    S = rng.normal(size=(N, L))
    W_o = ortho_group.rvs(p) * np.sqrt(p)
    W = W_o[:L] * 3
    W[0] = 3.0 * W[0]

    X_hat = np.dot(S, W)
    X_noise = sigma * rng.normal(size=(N, p))
    X = X_hat + X_noise

    logits = phi * S[:, 0] * 5
    probs = np.exp(logits) / (1 + np.exp(logits))
    y = binom.rvs(1, probs, random_state=42)
    return X, y, X_hat, S, W, logits


def test_ppca():
    X, y, X_hat, S, W, logits = PPCA_generate_data(L=3, p=200)
    training_options = {"n_iterations": 1000}
    model = PPCAAdversarial(
        2, mu=100.0, gamma=10.0, training_options=training_options
    )
    model.fit(X, y)
    S_ = model.transform(X)
