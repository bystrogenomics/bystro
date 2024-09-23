import numpy as np
import numpy.linalg as la
from bystro.supervised_ppca.ppca_augmented import (
    PPCAadversarial,
    PPCAsupervised,
    PPCASupAdversarial,
    PPCAadversarialRandomized,
)


def cosine_similarity(vec1, vec2):
    v1 = np.squeeze(vec1)
    v2 = np.squeeze(vec2)
    num = np.dot(v1, v2)
    denom = la.norm(v1) * la.norm(v2)
    return num / denom


def test_initialization():
    """Test the initialization of the PPCAadversarial class."""
    pca_model = PPCAadversarial(n_components=2)
    assert (
        pca_model.n_components == 2
    ), "n_components should be initialized correctly."


def test_fit_adversarial():
    """Test the fitting process of the PPCAadversarial model."""
    # Create a synthetic dataset
    rng = np.random.default_rng(2021)
    L = 3
    p = 100
    N = 1000
    S = rng.normal(size=(N, L))
    W = rng.normal(size=(L, p))
    W[0] = 3 * W[0]
    W[1] = 2 * W[1]
    X_hat = np.dot(S, W)
    X = X_hat + rng.normal(size=(N, p))

    Y = S[:, 0].reshape((-1, 1))

    pca_model = PPCAadversarial(n_components=2, mu=10000.0)
    pca_model.fit(X, Y)

    assert pca_model.W_ is not None, "W_ should be set after fitting."
    assert pca_model.sigma2_ is not None, "sigma2_ should be set after fitting."

    # Ensure that the learned components are orthogonal to the
    # sensitive information.
    assert np.abs(cosine_similarity(pca_model.W_[0], W[0])) < 0.1
    assert np.abs(cosine_similarity(pca_model.W_[1], W[0])) < 0.1

    pca_model = PPCAadversarial(
        n_components=2, mu=10000.0, regularization="Empirical"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAadversarial(
        n_components=2, mu=10000.0, regularization="Linear"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAadversarial(
        n_components=2, mu=10000.0, regularization="LinearInverse"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAadversarial(
        n_components=2, mu=10000.0, regularization="QuadraticInverse"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAadversarial(
        n_components=2, mu=10000.0, regularization="GeometricInverse"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAadversarial(
        n_components=2, mu=10000.0, regularization="NonLinear"
    )
    pca_model.fit(X, Y)


def test_fit_randomized():
    rng = np.random.default_rng(2021)
    L = 3
    p = 100
    N = 1000
    S = rng.normal(size=(N, L))
    W = rng.normal(size=(L, p))
    W[0] = 3 * W[0]
    W[1] = 2 * W[1]
    X_hat = np.dot(S, W)
    X = X_hat + rng.normal(size=(N, p))

    Y = S[:, 0].reshape((-1, 1))

    pca_model = PPCAadversarialRandomized(n_components=2, mu=10000.0)
    pca_model.fit(X, Y)


def test_fit_supervised():
    """Test the fitting process of the PPCAsupervised model."""
    # Create a synthetic dataset
    rng = np.random.default_rng(2021)
    L = 3
    p = 100
    N = 1000
    S = rng.normal(size=(N, L))
    W = rng.normal(size=(L, p))
    W[0] = 3 * W[0]
    W[1] = 2 * W[1]
    X_hat = np.dot(S, W)
    X = X_hat + rng.normal(size=(N, p))

    Y = S[:, 2].reshape((-1, 1))

    pca_model = PPCAsupervised(n_components=2, mu=10000.0)
    pca_model.fit(X, Y)

    assert pca_model.W_ is not None, "W_ should be set after fitting."
    assert pca_model.sigma2_ is not None, "sigma2_ should be set after fitting."

    # Ensure that the learned components are orthogonal to the
    # sensitive information.
    assert np.abs(cosine_similarity(pca_model.W_[0], W[2])) > 0.8

    pca_model = PPCAsupervised(
        n_components=2, mu=10000.0, regularization="Empirical"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAsupervised(
        n_components=2, mu=10000.0, regularization="Linear"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAsupervised(
        n_components=2, mu=10000.0, regularization="LinearInverse"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAsupervised(
        n_components=2, mu=10000.0, regularization="QuadraticInverse"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAsupervised(
        n_components=2, mu=10000.0, regularization="GeometricInverse"
    )
    pca_model.fit(X, Y)

    pca_model = PPCAsupervised(
        n_components=2, mu=10000.0, regularization="NonLinear"
    )
    pca_model.fit(X, Y)


def test_fit_supervised_adversarial():
    """Test the fitting process of the PPCAadversarial model."""
    # Create a synthetic dataset
    rng = np.random.default_rng(2021)
    L = 4
    p = 300
    N = 10000
    S = rng.normal(size=(N, L))
    W = rng.normal(size=(L, p))
    W[0] = 4 * W[0]
    W[1] = 3 * W[1]
    W[2] = 2 * W[1]
    X_hat = np.dot(S, W)
    X = X_hat + rng.normal(size=(N, p))

    Y = S[:, 3].reshape((-1, 1))
    Z = S[:, 0].reshape((-1, 1))

    pca_model = PPCASupAdversarial(n_components=2, mu_s=10000.0, mu_a=100000.0)
    pca_model.fit(X, Y, Z)

    assert pca_model.W_ is not None, "W_ should be set after fitting."
    assert pca_model.sigma2_ is not None, "sigma2_ should be set after fitting."

    assert np.abs(cosine_similarity(pca_model.W_[0], W[0])) < 0.2
    assert np.abs(cosine_similarity(pca_model.W_[1], W[0])) < 0.2

    assert np.abs(cosine_similarity(pca_model.W_[1], W[3])) > 0.8

    pca_model = PPCASupAdversarial(
        n_components=2, mu_s=10000.0, regularization="Empirical"
    )
    pca_model.fit(X, Y, Z)

    pca_model = PPCASupAdversarial(
        n_components=2, mu_s=10000.0, regularization="Linear"
    )
    pca_model.fit(X, Y, Z)

    pca_model = PPCASupAdversarial(
        n_components=2, mu_s=10000.0, regularization="LinearInverse"
    )
    pca_model.fit(X, Y, Z)

    pca_model = PPCASupAdversarial(
        n_components=2, mu_s=10000.0, regularization="QuadraticInverse"
    )
    pca_model.fit(X, Y, Z)

    pca_model = PPCASupAdversarial(
        n_components=2, mu_s=10000.0, regularization="GeometricInverse"
    )
    pca_model.fit(X, Y, Z)

    pca_model = PPCASupAdversarial(
        n_components=2, mu_s=10000.0, regularization="NonLinear"
    )
    pca_model.fit(X, Y, Z)
