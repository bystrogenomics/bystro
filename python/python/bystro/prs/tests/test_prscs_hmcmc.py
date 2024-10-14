import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from bystro.prs.prscs_hmcmc import PrsCSData


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
    X, y = generate_data_prscs(N=100, p=5)
    model = PrsCSData(mcmc_options={"num_samples": 10, "warmup_steps": 1})
    try:
        model.fit(X.astype(np.float32), y.astype(np.float32))
    except Exception as e:
        pytest.fail(f"fit method raised na exception: {e}")
