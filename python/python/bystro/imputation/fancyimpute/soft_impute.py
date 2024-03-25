"""
Soft Impute Algorithm for Matrix Completion

The SoftImpute algorithm addresses the problem of filling in missing 
entries of a matrix, a common issue in data analysis, particularly in the 
fields of collaborative filtering, data integration, and bioinformatics. 
This technique is particularly suited for scenarios where the data matrix 
is assumed to have a low-rank structure, meaning that the data can be 
approximated well by a matrix with fewer dimensions.

At its core, SoftImpute iteratively replaces missing values with values 
derived from a singular value decomposition (SVD) of the observed entries. 
The algorithm can be broken down into a few key steps:

1. Initialization: The missing entries of the input matrix are initially 
   filled with some default value, typically zeros.

2. Singular Value Decomposition (SVD): The algorithm performs SVD on the 
   filled matrix to decompose it into its singular vectors and singular 
   values. SVD is a technique that factors a matrix into one matrix with 
   orthogonal columns, one diagonal matrix with non-negative entries 
   (the singular values), and another orthogonal matrix.

3. Shrinkage: The singular values obtained from SVD are then subjected to 
   a shrinkage operation, which reduces each singular value by a fixed 
   amount (the shrinkage value), but not allowing them to go below zero. 
   This step effectively denoises the data by reducing the impact of 
   less significant components.

4. Matrix Reconstruction: Using the shrunk singular values and the 
   original singular vectors, a low-rank approximation of the matrix is 
   reconstructed.

5. Iteration: Steps 2-4 are repeated, each time using the newly imputed 
   values from the previous iteration, until the change in the matrix 
   between iterations falls below a predetermined threshold, indicating 
   convergence.

6. Finalization: Once convergence is achieved, the algorithm returns the 
   imputed matrix.

SoftImpute leverages the idea that the underlying structure of the data 
can be captured by a few dimensions (low-rank approximation), making it 
powerful for imputing missing data in large matrices efficiently and 
effectively. The algorithm is versatile and can be adapted to various 
types of data and applications.

This implementation of SoftImpute extends the BaseImpute abstract base 
class, providing a concrete implementation of matrix completion using 
the SoftImpute algorithm.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.linalg as la
from sklearn.utils.extmath import randomized_svd
from tqdm import trange

from bystro.imputation._base import BaseImpute

F32PREC = np.finfo(np.float32).eps


class SoftImpute(BaseImpute):
    """
    Implementation of the SoftImpute algorithm for matrix completion
    and imputation.

    This algorithm iteratively fills missing values in a matrix using
    soft-thresholded singular value decomposition (SVD).

    Parameters
    ----------
    shrinkage_value : float, default=0.1
        The shrinkage value or bias applied during the soft-
        thresholding step.
    training_options : Optional[Dict[str, Any]], default=None
        A dictionary containing options for the training process.
        Keys include 'n_iterations', 'convergence_threshold',
        'n_power_iterations', and 'max_rank'.
    init_fill_method : str, default="zero"
        The initial method to fill missing values before the
        iterative process begins. Options are 'zero', 'mean', and 'median'.
    """

    def __init__(
        self,
        shrinkage_value: float = 0.1,
        training_options: Optional[Dict[str, Any]] = None,
        init_fill_method: str = "zero",
    ) -> None:
        if training_options is None:
            training_options = {}
        super().__init__(
            fill_method=init_fill_method,
            training_options=training_options,
        )
        self.shrinkage_value = shrinkage_value

    def _converged(
        self, X_old: np.ndarray, X_new: np.ndarray, missing_mask: np.ndarray
    ) -> bool:
        """
        Check if the imputation has converged.

        Parameters
        ----------
        X_old : np.ndarray
            The imputed matrix from the previous iteration.
        X_new : np.ndarray
            The imputed matrix from the current iteration.
        missing_mask : np.ndarray
            A boolean array where True indicates a missing value in
            the original matrix.

        Returns
        -------
        bool
            True if the algorithm has converged, False otherwise.
        """
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference**2)
        old_norm = np.sqrt((old_missing_values**2).sum())
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        return (np.sqrt(ssd) / old_norm) < self.training_options[
            "convergence_threshold"
        ]

    def _svd_step(
        self,
        X: np.ndarray,
        shrinkage_value: float,
        max_rank: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Perform the singular value decomposition (SVD) step with soft
        thresholding.

        Parameters
        ----------
        X : np.ndarray
            The current matrix to decompose.
        shrinkage_value : float
            The value used for soft thresholding.
        max_rank : Optional[int], default=None
            The maximum rank of the decomposition.

        Returns
        -------
        Tuple[np.ndarray, int]
            The reconstructed matrix after applying the soft threshold,
            and the rank of the thresholded matrix.
        """
        training_options = self.training_options
        if max_rank:
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                n_iter=training_options["n_power_iterations"],
                random_state=None,
            )
        else:
            (U, s, V) = la.svd(X, full_matrices=False, compute_uv=True)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)
        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
        return X_reconstruction, rank

    def _solve(
        self, X: np.ndarray, missing_mask: np.ndarray, progress_bar: bool = True
    ) -> np.ndarray:
        """
        Solve the imputation problem using SoftImpute algorithm.

        Parameters
        ----------
        X : np.ndarray
            The initial matrix with missing values.
        missing_mask : np.ndarray
            A boolean array where True indicates a missing value in the original matrix.
        progress_bar : bool, default=True
            Whether to show progress during the iteration.

        Returns
        -------
        np.ndarray
            The matrix with imputed values.
        """
        training_options = self.training_options

        X_filled = X

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            X_reconstruction, rank = self._svd_step(
                X_filled,
                self.shrinkage_value,
                max_rank=training_options["max_rank"],
            )

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask,
            )
            X_filled[missing_mask] = X_reconstruction[missing_mask]
            if converged:
                break

        return X_filled

    def _fill_training_options(
        self, training_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates and fills the training options with default values.

        Parameters
        ----------
        training_options : Dict[str, Any]
            The user-provided training options:
            n_iterations : number of iterations of algorithm
            convergence_threshold : criteria for ending training if
                                    only small change observed
            n_power_iterations : number of iterations for randomized SVD
            max_rank : maximum rank of decomposition

        Returns
        -------
        Dict[str, Any]
            The validated and completed training options.

        Raises
        ------
        ValueError
            If there are missing expected options or unrecognized
            options provided.
        """
        default_options = {
            "n_iterations": 100,
            "convergence_threshold": 0.001,
            "n_power_iterations": 1,
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
