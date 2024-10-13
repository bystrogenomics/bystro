import numpy as np
from sklearn.preprocessing import StandardScaler
from bystro.prs.prscs_sgld import PRSCS
from sklearn.linear_model import Ridge


def generate_data_prscs(N=100000, p=25, sigma=np.sqrt(0.1)):
    rng = np.random.default_rng(2021)
    Z1 = rng.binomial(1, 0.1, size=(N, p))
    Z2 = rng.binomial(1, 0.1, size=(N, p))
    Z = Z1 + Z2
    Z_s = StandardScaler().fit_transform(Z)
    beta = rng.normal(0, 1, size=p)
    Zb = np.dot(Z_s, beta)
    eps = rng.normal(0, sigma, size=N)
    y = Zb + eps
    y = (y - np.mean(y)) / np.std(y)
    return Z_s, y


def test_prscs():
    X, y = generate_data_prscs()
    mm = Ridge()
    mm.fit(X, y)
    beta = np.squeeze(mm.coef_)
    model = PRSCS(training_options={"n_samples": 5000, "batch_size": 100})
    model.fit(X, y,progress_bar=False)

    posterior_mean = np.mean(model.samples_beta[2500:], axis=0)
    assert np.mean((posterior_mean - beta) ** 2) < 0.1
