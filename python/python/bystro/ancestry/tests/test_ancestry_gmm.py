import pytest
import numpy as np
from bystro.ancestry.gmm_ancestry import GaussianMixturePPCA


@pytest.fixture
def example_data():
    # Provide an example dataset for testing
    rng = np.random.default_rng(2021)

    return rng.normal(
        size=(100, 10)
    )  # Example data with 100 samples and 10 features


def test_gaussian_mixture_ppca_fit(example_data):
    # Test the fit method of GaussianMixturePPCA

    # Create an instance of the model with minimal configuration for testing
    model = GaussianMixturePPCA(n_clusters=2, n_components=5)

    # Perform fit on the example data
    model.fit(example_data, progress_bar=False)

    # Assert that the model has been trained (you can add more specific assertions)
    assert hasattr(model, "W_")
    assert hasattr(model, "sigma2_")
    assert hasattr(model, "pi_")
    assert hasattr(model, "mu_")


"""
def test_gaussian_mixture_ppca_transform(example_data):
    # Test the transform method of GaussianMixturePPCA

    # Create an instance of the model with minimal configuration for testing
    model = GaussianMixturePPCA(n_clusters=2, n_components=5)

    # Perform fit on the example data
    model.fit(example_data, progress_bar=False)

    # Perform transformation on the example data
    transformed_data = model.transform(example_data)

    # Assert that the transformed_data has the correct shape
    assert transformed_data.shape == (len(example_data), model.n_components)
"""
