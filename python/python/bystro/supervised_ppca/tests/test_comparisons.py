import numpy as np
import numpy.linalg as la

from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from scipy.stats import ortho_group  # type: ignore
from scipy.stats import binom  # type: ignore

import torch
import pytest

from bystro.supervised_ppca.gf_comparisons import (
    kl_divergence_gaussian,
    PPCADropoutVAE,
    PPCASVAE,
)


def mvn_kl_divergence_np(Sigma1, Sigma2, mu1, mu2):
    """
    Computes the kl divergence between two gaussian distributions.

    p ~ N(mu1,S1)
    q ~ N(mu2,S2)

    KL(p|q)

    Parameters
    ----------
    Sigma1 : np.array-like(p,p)
        First covariance matrix

    Sigma2 : np.array-like(p,p)
        Second covariance matrix

    mu1 : np.array-like(p,)
        First mean (optional)

    mu2 : np.array-like(p,)
        Second mean (optional)

    Returns
    -------
    kl_divergence : float
        Divergence between two distributions
    """
    p = Sigma1.shape[0]
    _, logdet2 = la.slogdet(Sigma2)
    _, logdet1 = la.slogdet(Sigma1)
    term1 = logdet2 - logdet1
    term2 = -p
    term3 = np.trace(la.solve(Sigma2, Sigma1))
    kl_divergence = term1 + term2 + term3
    diff = np.squeeze(mu1 - mu2)
    end = la.solve(Sigma2, diff)
    term4 = np.dot(diff, end)
    kl_divergence += term4
    kl_divergence *= 0.5
    return kl_divergence


def test_kl_divergence_gaussian():
    """
    Test the KL divergence computation between two multivariate Gaussian distributions.
    """
    d = 2  # Dimensionality
    mu0 = torch.zeros(d)
    sigma0 = torch.eye(d)
    mu1 = torch.tensor([0.0, 0.0])
    sigma1 = torch.eye(d) * 2

    # For Gaussians: P ~ N(0, I), Q ~ N(0, 2I), the KL divergence should be:
    expected_kl_div = 0.5 * (
        2 * torch.log(torch.tensor(2.0)) - d + (1 / 2) * d - 0.5 * 0
    )

    calculated_kl_div = kl_divergence_gaussian(mu0, sigma0, mu1, sigma1)

    # Assert that the calculated KL divergence is close to the expected value.
    # Using pytest's approx for comparison due to potential
    # floating-point arithmetic issues.
    assert calculated_kl_div.item() == pytest.approx(expected_kl_div.item())


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


def test_vae_dropout():
    X, y, X_hat, S, W, logits = PPCA_generate_data(L=3, p=200)
    training_options = {"n_iterations": 1000}
    model = PPCADropoutVAE(
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

    training_options = {"n_iterations": 1000, "use_gpu": False}
    model = PPCADropoutVAE(
        2, mu=100.0, gamma=10.0, delta=5.0, training_options=training_options
    )
    model.fit(X, y)


def test_svae():
    X, y, X_hat, S, W, logits = PPCA_generate_data(L=3, p=200)
    training_options = {"n_iterations": 10, "learning_rate": 1e-4}
    model = PPCASVAE(
        2,
        mu=1.0,
        gamma=10.0,
        delta=5.0,
        lamb=10.0,
        training_options=training_options,
    )
    model.fit(X, y)

    training_options = {
        "n_iterations": 10,
        "use_gpu": False,
        "learning_rate": 1e-4,
    }
    model = PPCASVAE(
        2,
        mu=10.0,
        gamma=10.0,
        delta=5.0,
        lamb=10.0,
        training_options=training_options,
    )
    model.fit(X, y)
    model.transform(X)
    model.transform_encoder(X)
