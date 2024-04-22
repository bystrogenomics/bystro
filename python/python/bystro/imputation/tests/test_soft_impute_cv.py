import pytest
import numpy as np
from bystro.imputation.soft_impute_cv import (
    nan_with_probability,
    SoftImputeCV,
)  # Replace 'your_module' with the actual module name


def test_nan_with_probability():
    rng = np.random.default_rng(42)
    X = np.ones((100,10))
    p = 0.3
    X_modified, mask = nan_with_probability(X, p, rng)

    assert (
        X.shape == X_modified.shape
    ), "Shape of modified array should be the same as input"
    assert mask.dtype == bool, "Mask should be of boolean type"
    assert np.all(
        np.isnan(X_modified) == mask
    ), "Mask should correctly indicate NaN placements"


def test_SoftImputeCV_initialization():
    cv = SoftImputeCV(Cs=5, seed=42, k_fold=5, prob_holdout=0.1)
    assert (
        cv.Cs == 5
        and cv.seed == 42
        and cv.k_fold == 5
        and cv.prob_holdout == 0.1
    ), "Attributes should be correctly set"


@pytest.fixture
def setup_data():
    # Create a dataset with some missing values
    rng = np.random.default_rng(2021)
    X = rng.uniform(size=(100, 100))
    X[0, 0] = np.nan
    return X


def test_fit_transform(setup_data):
    cv = SoftImputeCV(Cs=1, seed=42, k_fold=2, prob_holdout=0.05)
    result = cv.fit_transform(setup_data)
    assert not np.isnan(
        result
    ).all(), "Output should have imputed values where there were NaNs"


# Run tests with pytest
if __name__ == "__main__":
    pytest.main()
