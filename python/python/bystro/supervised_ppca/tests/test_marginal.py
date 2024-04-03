import pytest
import numpy as np
from bystro.supervised_ppca.gf_marginal import PPCAMarginal


@pytest.fixture
def basic_ppca_marginal_setup():
    """Fixture for creating a basic PPCAMarginal instance and related data."""
    n_components = 2
    gamma = 10.0
    training_options = {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "n_iterations": 100,
    }
    ppcam = PPCAMarginal(
        n_components=n_components,
        gamma=gamma,
        training_options=training_options,
    )
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(100, 5)).astype(np.float32)
    idx_list = [
        np.array([True, True, False, False, True]),
        np.array([True, False, True, True, False]),
    ]
    lamb = np.array([2.0, 3.0])
    return ppcam, X, idx_list, lamb


def test_initialization():
    """Test the initialization of PPCAMarginal."""
    ppcam = PPCAMarginal()
    assert ppcam.n_components == 2
    assert ppcam.gamma == 10.0


def test_fit(basic_ppca_marginal_setup):
    """Test fitting PPCAMarginal with example data."""
    ppcam, X, idx_list, lamb = basic_ppca_marginal_setup
    ppcam.fit(X, idx_list, lamb)
    assert hasattr(
        ppcam, "W_"
    ), "Model wasn't fitted properly, 'W_' attribute missing."


def test_invalid_inputs():
    """Test PPCAMarginal with invalid inputs to ensure proper error handling."""
    ppcam = PPCAMarginal()
    with pytest.raises(ValueError):
        # Assuming _test_inputs will be invoked and catch these invalid inputs
        ppcam.fit(
            X=np.array([[1, 2], [3, 4]]),
            idx_list=[np.array([True, False])],
            lamb=np.array([1.0, 2.0]),
        )
