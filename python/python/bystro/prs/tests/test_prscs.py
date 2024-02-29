import numpy as np
from sklearn.preprocessing import StandardScaler
from bystro.prs.prscs import PRSCS


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
    return Z_s, y, beta


def test_prscs():
    X, y, beta = generate_data_prscs()
    model = PRSCS(training_options={'n_samples':10000,'batch_size':1000})
    model.fit(X, y)

    posterior_mean = np.mean(model.samples_beta[100:], axis=0)
    print(beta)
    print('>>>>>>>>>>>>')
    print(posterior_mean)
    print(model.samples_sigma2.shape)
    assert np.mean((posterior_mean - beta) ** 2) < 0.1
