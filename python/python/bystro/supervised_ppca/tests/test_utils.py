import numpy as np
from bystro.supervised_ppca._misc_np import classify_missingness


def test_classify_missingness():
    # Test Case 1
    input_matrix_1 = np.array(
        [
            [1, 2, np.nan, 4],
            [5, np.nan, np.nan, 8],
            [4, np.nan, np.nan, 8],
            [5, np.nan, np.nan, np.nan],
            [4, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 8],
            [np.nan, np.nan, np.nan, 7],
            [9, 10, 11, 12],
            [9, 10, 11, 13],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    matrices_list_1, vectors_list_1 = classify_missingness(input_matrix_1)

    assert len(matrices_list_1) == 6
    assert len(vectors_list_1) == 6
