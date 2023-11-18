import pytest # type: ignore
import numpy as np
from bystro.rare_variant.markov_random_field import MarkovRandomFieldNCE


@pytest.fixture
def mrf_model():
    prior_options = {
        "mu_phi": -2.0,
        "sigma_phi": 1.0,
        "mu_L_l": -1.0,
        "sigma_L_l": 1.0,
    }
    training_options = {
        "n_iterations": 100,
        "learning_rate": 1e-3,
        "batch_size": 10,
        "nu": 5,
    }
    return MarkovRandomFieldNCE(
        prior_options=prior_options, training_options=training_options
    )


def test_fit(mrf_model):
    # Generate dummy data for testing
    rng = np.random.default_rng(2021)
    n_samples = 50
    n_covariates = 10
    X = rng.normal(size=(n_samples, n_covariates))

    # Ensure the fit method runs without errors
    mrf_model.fit(X, progress_bar=False)

    # Add more specific assertions based on the expected behavior of your code
    assert mrf_model.Phi_ is not None
    assert mrf_model.Theta_ is not None
    assert mrf_model.log_z_ is not None


def test_score(mrf_model):
    # Generate dummy data for testing
    rng = np.random.default_rng(2021)
    n_samples = 50
    n_covariates = 10
    X = rng.normal(size=(n_samples, n_covariates))

    # Ensure the score method runs without errors
    score = mrf_model.score(X)

    # Add more specific assertions based on the expected behavior of your code
    assert isinstance(score, float)


def test_score_samples(mrf_model):
    # Generate dummy data for testing
    rng = np.random.default_rng(2021)
    n_samples = 50
    n_covariates = 10
    X = rng.normal(size=(n_samples, n_covariates))

    # Ensure the score_samples method runs without errors
    sample_scores = mrf_model.score_samples(X)

    # Add more specific assertions based on the expected behavior of your code
    assert len(sample_scores) == n_samples
    assert all(isinstance(score, float) for score in sample_scores)
