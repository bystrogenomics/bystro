"""
This module provides an implementation of the SoftImpute algorithm
with automatic cross-validation to select the optimal regularization
strength. The `SoftImputeCV` class handles the model fitting and
transformation process, using internal cross-validation to determine
the best regularization parameter from a set of possible values.

The SoftImpute algorithm imputes missing entries in a matrix based
on low-rank SVD by iteratively replacing missing values with estimates
and updating the singular value decomposition of the matrix.

Classes
-------
- SoftImputeCV : Automatically tunes the regularization strength of
  SoftImpute using cross-validation.

Methods
-------
- fit_transform : Fits the model to the data and transforms the input
  matrix by imputing missing values.

Mathematical Description:
- Given a matrix X with missing entries, SoftImpute minimizes the
  objective function ||X - XY||_F^2 + lambda * ||Y||_*, where ||.||_F
  is the Frobenius norm, ||.||_* is the nuclear norm, and lambda is the
  regularization parameter.

References:
- Mazumder, R., Hastie, T. & Tibshirani, R. (2010). Spectral
  Regularization Algorithms for Learning Large Incomplete Matrices,
  JMLR.
- Hastie, T., Mazumder, R., Lee, J. D., & Zadeh, R. (2015). Matrix
  Completion and Low-Rank SVD via Fast Alternating Least Squares,
  JMLR.
"""
from typing import Any, Dict
import numpy as np

from bystro.imputation.fancyimpute.soft_impute import SoftImpute


def nan_with_probability(X, p, rng):
    """
    Takes an input array X and a float p.
    Of the values in X that are not NaN, sets the elements to NaN with
    probability p. Returns the modified array X and a mask indicating
    which values were set to NaN.

    Parameters
    ----------
    X: np.array,
        the input array possibly containing NaNs.

    p: float,
        probability with which a non-NaN element of X is set to NaN.

    rng : numpy random generator
        The random number generator

    Returns
    -------
    X_modified: np.array,
        the modified array with additional NaNs.

    mask: np.array,
        a boolean array where True indicates a value was set to
        NaN by this function.
    """
    X = X.copy()
    # Create a mask of the same shape as X, initially all False
    mask = np.zeros(X.shape, dtype=bool)
    # Find indices of non-NaN values in X
    non_nan_indices = np.where(~np.isnan(X))
    # Generate random numbers for these indices
    random_numbers = rng.uniform(low=0, high=1, size=len(non_nan_indices[0]))
    # Determine which values to set to NaN based on probability p
    to_nan = random_numbers < p
    # Apply the decision to X and mask
    X[non_nan_indices[0][to_nan], non_nan_indices[1][to_nan]] = np.nan
    mask[non_nan_indices[0][to_nan], non_nan_indices[1][to_nan]] = True
    return X, mask


class SoftImputeCV:
    """
    Cross-validation for SoftImpute algorithm which automatically selects 
    the regularization strength that leads to the best data reconstruction. 
    This is achieved by creating a small hold-out set from the observed 
    data and testing how well each candidate regularization strength can 
    impute these "missing" values.

    Parameters
    ----------
    Cs : int, default=10
        The number of regularization strengths to consider. Regularization 
        strengths are logarithmically spaced between 10^-2 and 10^2.
    seed : int, default=2021
        Random seed for reproducibility.
    k_fold : int, default=3
        The number of folds in the cross-validation.
    prob_holdout : float, default=0.05
        The proportion of the observed data to hold out for validation.
    training_options : dict, optional
        Additional options to pass to the imputation model during training.

    Attributes
    ----------
    training_options : dict
        Stores training options after processing.
    """
    def __init__(
        self,
        Cs=10,
        seed=2021,
        k_fold=3,
        prob_holdout=0.05,
        training_options=None,
    ):
        self.Cs = Cs
        self.seed = seed
        self.k_fold = k_fold
        self.prob_holdout = prob_holdout
        if training_options is None:
            training_options = {}
        self.training_options = self._fill_training_options(training_options)

    def fit_transform(self, X):
        """
        Fits the imputation model on the input data matrix X using 
        cross-validated selection of the regularization strength and 
        then transforms the data by imputing missing values.

        Parameters
        ----------
        X : ndarray
            The data matrix to be imputed, where NaNs represent 
            missing values.

        Returns
        -------
        X_imputed : ndarray
            The imputed data matrix with no missing values.
        """
        rng = np.random.default_rng(self.seed)
        regs = np.logspace(-2, 2, self.Cs)  # Regularization strengths

        imputation_var = np.zeros((self.Cs, self.k_fold))

        for i in range(self.Cs):
            for j in range(self.k_fold):
                X_modified, mask = nan_with_probability(
                    X.copy(), self.prob_holdout, rng
                )
                model_si = SoftImpute(
                    shrinkage_value=regs[i],
                    training_options=self.training_options,
                )
                X_complete_si = model_si.fit_transform(X_modified)

                vals_imputed_si = X_complete_si[mask]
                vals_original = X[mask]
                random_guess_mse = np.mean(
                    (vals_original - np.mean(vals_original)) ** 2
                )
                si_mse = np.mean((vals_original - vals_imputed_si) ** 2)
                imputation_var[i, j] = 1 - si_mse / random_guess_mse

        avg_var = np.mean(imputation_var, axis=1)
        opt_val = regs[avg_var == np.amax(avg_var)]
        model_si = SoftImpute(
            shrinkage_value=opt_val, training_options=self.training_options
        )
        X_complete = model_si.fit_transform(X)

        return X_complete

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
