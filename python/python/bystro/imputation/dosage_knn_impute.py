from typing import Any, Dict, Optional, Tuple
from sklearn.utils import check_array # type: ignore

import numpy as np
from tqdm import trange

from bystro.imputation._base import BaseImpute


class KNNDosage(BaseImpute):
    def __init__(
        self,
        n_neighbors: int = 5,
        width: int = 5,
        training_options: Optional[Dict[str, Any]] = None,
        init_fill_method: str = "zero",
    ) -> None:
        if training_options is None:
            training_options = {}
        super().__init__(
            fill_method=init_fill_method,
            training_options=training_options,
        )
        self.n_neighbors = n_neighbors
        self.width = width

    def _fill_training_options(
        self, training_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Abstract method to be implemented by subclasses for handling additional training options.

        Parameters
        ----------
        training_options : Dict[str, Any]
            A dictionary of training options.

        Returns
        -------
        Dict[str, Any]
            A dictionary of processed training options.
        """
        default_options = {
            "max_rank": None,
        }
        tops = {**default_options, **training_options}

        default_keys = set(default_options.keys())
        final_keys = set(tops.keys())

        expected_but_missing_keys = default_keys - final_keys
        unexpected_but_present_keys = final_keys - default_keys
        if expected_but_missing_keys:
            raise ValueError("training options were expected but not found")
        if unexpected_but_present_keys:
            raise ValueError("training options were unrecognized but provided")

        return tops

    def _solve(self, X, missing_mask, progress_bar=True):
        """
        Abstract method to be implemented by subclasses for solving the imputation problem.

        Parameters
        ----------
        X : np.ndarray
            The data array with missing values filled in.
        missing_mask : np.ndarray
            A boolean mask indicating the positions of originally missing values in `X`.
        progress_bar : bool, default=True
            Whether to display a progress bar during the operation (if applicable).

        Returns
        -------
        np.ndarray
            The data array with missing values imputed.
        """
        self._test_inputs(X)
        validate_matrix_values(X)
        N, p = X.shape
        X_imputed = X.copy()


        for i in trange(N,disable=not progress_bar):
            for j in range(p):
                if missing_mask[i, j]:
                    # Extract the subset matrix
                    start_idx = max(0, j - self.width)
                    end_idx = min(p, j + self.width + 1)
                    X_sub = X[:, start_idx:end_idx]
                    mask_sub = missing_mask[:, start_idx:end_idx]

                    picked_value = j - start_idx

                    valid_rows = np.logical_not(missing_mask[:, j])
                    X_sub_sub = X_sub[valid_rows]
                    X_curr = X_sub[i]
                    mask_curr = missing_mask[i, start_idx:end_idx]
                    mask_sub_sub = mask_sub[valid_rows]

                    # Compute Hamming distances, ignoring missing values
                    distances = []
                    for kk in range(X_sub_sub.shape[0]):
                        row_curr = X_sub_sub[kk]
                        mask_row = mask_sub_sub[kk]
                        mask_row_row = np.logical_not(
                            mask_curr
                        ) & np.logical_not(mask_row)
                        X_curr_sel = X_curr[mask_row_row]
                        X_sub_sel = row_curr[mask_row_row]
                        distances.append(np.sum(X_curr_sel != X_sub_sel))

                    # Sort by distance and select the closest n_neighbors
                    sorted_indices = np.argsort(distances)
                    X_sort = X_sub_sub[sorted_indices]
                    vals = X_sort[: self.n_neighbors, picked_value]
                    unique_vals, counts = np.unique(vals, return_counts=True)
                    max_count = np.max(counts)
                    most_common_values = unique_vals[counts == max_count]
                    most_common_value = np.min(most_common_values)
                    X_imputed[i, j] = most_common_value
        return X_imputed

    def _transform_training_data(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the training data by identifying missing values.

        Parameters
        ----------
        X : np.ndarray
            The data array to be transformed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the original data array `X` and a
            boolean mask `missing_mask` indicating the positions
            of missing values in `X`.
        """
        X = check_array(X, force_all_finite=False)
        missing_mask_nan = np.isnan(X)
        missing_mask_n1 = X == -1
        missing_mask = missing_mask_nan | missing_mask_n1

        return X, missing_mask


def validate_matrix_values(X):
    """
    Validate that the matrix X contains only the values 0, 1, 2, -1, or NaN.

    Parameters:
    X (numpy array): The input matrix to be validated.

    Raises:
    ValueError: If the matrix contains values other than 0, 1, 2, -1, or NaN.
    """
    allowed_values = {0, 1, 2, -1}

    non_nan_mask = ~np.isnan(X)
    unique_values = set(X[non_nan_mask].flatten())

    if not unique_values.issubset(allowed_values):
        invalid_values = unique_values.difference(allowed_values)
        raise ValueError(f"Matrix contains invalid values: {invalid_values}")
