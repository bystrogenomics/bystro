import numpy as np
from bystro.imputation.dosage_knn_impute import KNNDosage


def test_impute_missing_values():
    # Test case 1
    X1 = np.array([[0, 1, 2, 1], [0, -1, 1, 2], [2, 1, 0, 1], [-1, 0, 1, 2]])
    expected_X1 = np.array(
        [[0, 1, 2, 1], [0, 0, 1, 2], [2, 1, 0, 1], [0, 0, 1, 2]]
    )
    model = KNNDosage(n_neighbors=1)
    result = model.fit_transform(X1)
    assert np.sum(np.abs(expected_X1 - result)) == 0

    # Test case 2
    X2 = np.array(
        [[0, 1, 2, 1], [0, np.nan, 1, 2], [2, 1, 0, 1], [np.nan, 0, 1, 2]]
    )
    expected_X2 = np.array(
        [[0, 1, 2, 1], [0, 0, 1, 2], [2, 1, 0, 1], [0, 0, 1, 2]]
    )
    result = model.fit_transform(X2)
    assert np.sum(np.abs(expected_X2 - result)) == 0

    # Test case 3: All values are already filled
    X3 = np.array([[0, 1, 2, 1], [1, 0, 0, 2], [2, 1, 0, 1], [0, 0, 1, 2]])
    result = model.fit_transform(X3)
    assert np.sum(np.abs(X3 - result)) == 0

    X4 = np.array(
        [
            [0, 1, 2, 1],
            [0, 0, 1, 2],
            [2, 1, 0, -1],
            [2, 1, 0, 0],
            [2, 1, 0, 1],
            [2, 1, 0, 1],
            [0, 0, 1, 2],
        ]
    )
    expected_X4 = np.array(
        [
            [0, 1, 2, 1],
            [0, 0, 1, 2],
            [2, 1, 0, 1],
            [2, 1, 0, 0],
            [2, 1, 0, 1],
            [2, 1, 0, 1],
            [0, 0, 1, 2],
        ]
    )
    model = KNNDosage(n_neighbors=3)
    result = model.fit_transform(X4)
    print(result)
    assert np.sum(np.abs(expected_X4 - result)) == 0
