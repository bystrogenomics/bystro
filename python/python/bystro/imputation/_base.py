import abc
import numpy as np
from sklearn.utils import check_array
from typing import Any, Callable, Dict, Optional, Tuple


class BaseImpute(abc.ABC):
    """
    Abstract base class for imputation.

    This class provides a skeletal implementation of the imputation methods
    that can be extended by concrete imputation classes.

    Parameters
    ----------
    fill_method : str, default='zero'
        Specifies the method used to fill in the missing values. Options
        are 'zero', 'mean', and 'median'.
    training_options : Optional[Dict[str, Any]], default=None
        Additional options for training the imputation model.
    """

    def __init__(
        self,
        fill_method: str = "zero",
        training_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fill_method = fill_method
        if training_options is None:
            training_options = {}
        self.training_options = self._fill_training_options(training_options)

    def _fill_columns_with_fn(
        self,
        X: np.ndarray,
        missing_mask: np.ndarray,
        col_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Fills missing values in columns using a specified function.

        Parameters
        ----------
        X : np.ndarray
            The data array with missing values.
        missing_mask : np.ndarray
            A boolean mask indicating the positions of missing values in `X`.
        col_fn : Callable[[np.ndarray], np.ndarray]
            A function used to compute the fill value for each column. This
            function must accept a single column and return a fill value.
        """
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            X[missing_col, col_idx] = fill_values

    def _fill(
        self,
        X: np.ndarray,
        missing_mask: np.ndarray,
        fill_method: Optional[str] = None,
    ) -> np.ndarray:
        X = check_array(X, force_all_finite=False)

        X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method == "zero":
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        else:
            raise ValueError("Unrecognized fill method %s" % fill_method)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the imputer on `X` and returns the transformed array with missing values imputed.

        Parameters
        ----------
        X : np.ndarray
            The data array with missing values to impute.

        Returns
        -------
        np.ndarray
            The data array with missing values imputed.
        """
        self._test_inputs(X)
        X, missing_mask = self._transform_training_data(X)
        observed_mask = ~missing_mask

        X_filled = self._fill(X, missing_mask)
        X_result = self._solve(X_filled, missing_mask)
        X_result[observed_mask] = X[observed_mask]
        return X_result

    def _test_inputs(self, X: np.ndarray) -> None:
        """
        Tests input arrays for compatibility with the imputer.

        Parameters
        ----------
        X : np.ndarray
            The data array to test.

        Raises
        ------
        TypeError
            If `X` is not a NumPy array.
        ValueError
            If `X` is not a 2D array or contains completely unobserved rows or columns.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Expected NumPy array input but got %s" % (type(X)))
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

        X_mask = np.isnan(X)
        miss_column = np.mean(X_mask, axis=0)
        miss_row = np.mean(X_mask, axis=1)

        if np.any(miss_column == 1):
            raise ValueError("Entire column unobserved")
        if np.any(miss_row == 1):
            raise ValueError("Entire row unobserved")

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
        missing_mask = np.isnan(X)
        return X, missing_mask

    @abc.abstractmethod
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

    @abc.abstractmethod
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
