import numpy as np
from bystro.imputation.fancyimpute.nuclear_norm import NuclearNormMinimization


def reconstruction_error(XY, XY_completed, missing_mask):
    value_pairs = [
        (i, j, XY[i, j], XY_completed[i, j])
        for i in range(XY.shape[0])
        for j in range(XY.shape[1])
        if missing_mask[i, j]
    ]
    diffs = [actual - predicted for (_, _, actual, predicted) in value_pairs]
    missing_mse = np.mean([diff**2 for diff in diffs])
    missing_mae = np.mean([np.abs(diff) for diff in diffs])
    return missing_mse, missing_mae


def create_rank_k_dataset(
    n_rows=50,
    n_cols=50,
    k=3,
    fraction_missing=0.01,
    symmetric=False,
    random_seed=2021,
):
    rng = np.random.default_rng(random_seed)
    x = rng.normal(size=(n_rows, k))
    y = rng.normal(size=(k, n_cols))

    XY = np.dot(x, y)

    if symmetric:
        assert n_rows == n_cols
        XY = 0.5 * XY + 0.5 * XY.T

    missing_raw_values = rng.uniform(0, 1, size=(n_rows, n_cols))
    missing_mask = missing_raw_values < fraction_missing

    XY_incomplete = XY.copy()
    XY_incomplete[missing_mask] = np.nan

    return XY, XY_incomplete, missing_mask


def test_initialization():
    """Test initialization of the NuclearNormMinimization class with different parameters."""
    nnm = NuclearNormMinimization(
        require_symmetric_solution=True, init_fill_method="mean"
    )
    assert nnm.require_symmetric_solution 
    assert nnm.fill_method == "mean"


def test_solve_simple_case():
    """Test the _solve method with a simple case."""
    XY, XY_incomplete, missing_mask = create_rank_k_dataset()
    nnm = NuclearNormMinimization(
        training_options={"n_iterations": 10000, "convergence_threshold": 0.001}
    )
    XY_completed = nnm.fit_transform(XY_incomplete)
    assert isinstance(XY_completed, np.ndarray)
    _, missing_mae = reconstruction_error(XY, XY_completed, missing_mask)
    assert missing_mae < 0.1, "Error too high!"


def test_fill_training_options():
    """Test the handling of default and user-provided training options."""
    nnm = NuclearNormMinimization(training_options={"n_iterations": 50})
    expected_options = {
        "n_iterations": 50,
        "convergence_threshold": 0.001,
    }
    assert nnm.training_options == expected_options
