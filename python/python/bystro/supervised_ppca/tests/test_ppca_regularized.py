import numpy as np
from bystro.supervised_ppca.ppca_regularized import PPCARegularized


def test_initialization():
    """Test the initialization of the PPCARegularized class."""
    pca_model = PPCARegularized(n_components=2)
    assert (
        pca_model.n_components == 2
    ), "n_components should be initialized correctly."


def test_fit():
    """Test the fitting process of the PPCARegularized model."""
    # Create a synthetic dataset
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100, 5))  # 100 samples and 5 features

    pca_model = PPCARegularized(
        n_components=2,
        regularization_options={"method": "Bayesian", "prior_options": {}},
    )
    pca_model.fit(X)

    assert pca_model.W_ is not None, "W_ should be set after fitting."
    assert pca_model.sigma2_ is not None, "sigma2_ should be set after fitting."


def test_covariance_output():
    """Test the output of the covariance matrix computation."""
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100, 5))

    pca_model = PPCARegularized(
        n_components=2,
        regularization_options={"method": "Bayesian", "prior_options": {}},
    )
    pca_model.fit(X)

    cov_matrix = pca_model.get_covariance()
    assert cov_matrix.shape == (
        2,
        2,
    ), "Covariance matrix should have dimensions (n_components, n_components)."
    assert np.allclose(
        cov_matrix,
        np.dot(pca_model.W_.T, pca_model.W_)
        + pca_model.sigma2_ * np.eye(pca_model.p),
    ), "Covariance calculation error."


def test_noise_output():
    """Test the output of the noise matrix."""
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100, 5))

    pca_model = PPCARegularized(
        n_components=2,
        regularization_options={"method": "Bayesian"},
    )
    pca_model.fit(X)

    noise_matrix = pca_model.get_noise()
    assert noise_matrix.shape == (
        2,
        2,
    ), "Noise matrix should be (n_components, n_components)."
    assert np.all(
        noise_matrix == pca_model.sigma2_ * np.eye(pca_model.p)
    ), "Noise matrix calculation error."
