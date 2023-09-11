import numpy as np
import numpy.linalg as la
import scipy.stats as st  # type: ignore
from bystro.supervised_ppca.gf_generative_pt import PPCA


def generate_data_ppca():
    rng = np.random.default_rng(2021)
    N = 10000
    L = 4
    p = 50
    sigma = 1.0
    W_base = st.ortho_group.rvs(p)
    W = W_base[:L]
    lamb = rng.gamma(1, 1, size=L) + 1
    for ll in range(L):
        W[ll] = W[ll] * lamb[ll]
    S_train = rng.normal(size=(N, L))
    X_hat = np.dot(S_train, W)
    X_noise = sigma * rng.normal(size=(N, p))
    X_train = X_hat + X_noise
    return W, sigma, X_train, S_train


def generate_data_spca(L=4, sigmax=1.0, sigmay=0.5, N=10000):
    rng = np.random.default_rng(2021)
    p = 30
    q = 5
    W_x_base = st.ortho_group.rvs(p)
    W_y_base = st.ortho_group.rvs(q)
    Wx = W_x_base[:L]
    Wy = W_y_base[:L]
    lamb_x = np.sort(rng.gamma(1, 1, size=L))[::-1] + 1
    lamb_y = rng.gamma(1, 1, size=L) + 1
    for ll in range(L):
        Wx[ll] = Wx[ll] * lamb_x[ll]
        Wy[ll] = Wy[ll] * lamb_y[ll]
    S_train = rng.normal(size=(N, L))
    X_hat = np.dot(S_train, Wx)
    Y_hat = np.dot(S_train, Wy)
    X_noise = sigmax * rng.normal(size=(N, p))
    Y_noise = sigmay * rng.normal(size=(N, q))
    X_train = X_hat + X_noise
    Y_train = Y_hat + Y_noise
    return Wx, Wy, sigmax, sigmay, X_train, Y_train, S_train


def generate_data_factorAnalysis():
    rng = np.random.default_rng(2021)
    N = 100000
    L = 4
    p = 30
    alpha = 4
    beta = 4
    lamb = 0.25 * st.gamma(3, 1, 3).rvs(4)
    lamb = np.sort(lamb)
    lamb = lamb[::-1]
    S = rng.normal(size=(N, L))
    W_ = st.ortho_group.rvs(p)
    W_ = W_[:L]
    W_ = np.transpose(W_.T * lamb)
    W_ = 2 * W_
    ig = st.invgamma(alpha, 0, beta)
    phi = ig.rvs(p)  # *.01
    X_noise = rng.normal(size=(N, p)) * phi
    X_hat = np.dot(S, W_)
    X = X_hat + X_noise
    return X, S, W_, phi


def test_ppca():
    W, sigma, X, S_train = generate_data_ppca()
    model = PPCA(4, training_options={"n_iterations": 100})
    model.fit(X)
    cov_est = model.get_covariance()
    cov_emp = np.dot(X.T, X) / X.shape[0]
    s1 = la.norm(cov_emp - cov_est)
    assert s1 <= 10.0
