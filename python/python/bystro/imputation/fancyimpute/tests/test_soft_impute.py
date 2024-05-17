import numpy as np
from bystro.imputation.fancyimpute.soft_impute import SoftImpute


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


def test_soft_impute_with_low_rank_random_matrix():
    XY, XY_incomplete, missing_mask = create_rank_k_dataset()
    solver = SoftImpute()
    XY_completed = solver.fit_transform(XY_incomplete)
    _, missing_mae = reconstruction_error(XY, XY_completed, missing_mask)
    print(missing_mae)
    assert missing_mae < 0.1, "Error too high!"
