from numpy.random import default_rng
from bystro.supervised_ppca.gf_generative_missing_pt import PPCAM

# Set up the random number generator
rng = default_rng(seed=42)


def test_ppcam_fit():
    # Generate synthetic data for testing
    n_samples, n_covariates, n_components = 100, 5, 2
    X = rng.normal(size=(n_samples, n_covariates))

    ppcam = PPCAM(n_components=n_components, training_options={"n_iterations": 20})
    ppcam.fit(X)

    # Test if the model is fitted
    assert ppcam.W_ is not None
    assert ppcam.sigma2_ is not None
    assert ppcam.p is not None

    ppcam = PPCAM(
        n_components=n_components, training_options={"use_gpu": False, "n_iterations": 20}
    )
    ppcam.fit(X)


def test_ppcam_get_covariance():
    # Generate synthetic data for testing
    n_samples, n_covariates, n_components = 100, 5, 2
    X = rng.normal(size=(n_samples, n_covariates))

    ppcam = PPCAM(n_components=n_components, training_options={"n_iterations": 20})
    ppcam.fit(X)

    # Test get_covariance method
    covariance_matrix = ppcam.get_covariance()
    assert covariance_matrix.shape == (n_covariates, n_covariates)


def test_ppcam_get_noise():
    # Generate synthetic data for testing
    n_samples, n_covariates, n_components = 100, 5, 2
    X = rng.normal(size=(n_samples, n_covariates))

    ppcam = PPCAM(n_components=n_components, training_options={"n_iterations": 20})
    ppcam.fit(X)

    # Test get_noise method
    noise_matrix = ppcam.get_noise()
    assert noise_matrix.shape == (n_covariates, n_covariates)
