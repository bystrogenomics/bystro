import numpy as np
import numpy.linalg as la
from ..partial_least_squares_tf import PLS


def cosine_similarity_np(vec1, vec2):
    v1 = np.squeeze(vec1)
    v2 = np.squeeze(vec2)
    num = np.dot(v1, v2)
    denom = la.norm(v1) * la.norm(v2)
    return num / denom


def generate_data(Ls, Lx, eps=0.3):
    N = 1000000
    p = 60
    q = 20
    rng = np.random.default_rng(2021)
    B_true_x = rng.normal(size=(Lx, p)) * 0.5
    W_true_x = rng.normal(size=(Ls, p))
    W_true_y = rng.normal(size=(Ls, q))
    S_shared = rng.normal(size=(N, Ls))
    S_x = rng.normal(size=(N, Lx))
    X_hat_s = np.dot(S_shared, W_true_x)
    Y_hat_s = np.dot(S_shared, W_true_y)
    X_hat_x = np.dot(S_x, B_true_x)
    X_hat = X_hat_s + X_hat_x
    Y_hat = Y_hat_s

    X_noise = eps * rng.normal(size=(N, p))
    Y_noise = eps * rng.normal(size=(N, q))
    X = X_hat + X_noise
    Y = Y_hat + Y_noise
    V = np.hstack((X, Y))
    cov = np.cov(V.T)
    sigma = eps ** 2
    return X, Y, W_true_x, W_true_y, B_true_x, cov, sigma, Y_hat


def test_methods():
    Ls = 2
    Lx = 2
    X, Y, W_true_x, W_true_y, B_true_x, cov, sigma, Y_hat = generate_data(
        Ls, Lx
    )
    model = PLS(n_components=Ls, n_components_x=Lx)
    model.fit(X, y=Y)

    diff_s = np.abs(model.sigma2_ - sigma)
    assert diff_s <= 3e-1
