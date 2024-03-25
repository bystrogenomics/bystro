"""
Exact Matrix Completion via Convex Optimization

This module implements the matrix completion algorithm as described in 
the seminal work "Exact Matrix Completion via Convex Optimization" by 
Emmanuel Candes and Benjamin Recht. The algorithm addresses the problem 
of estimating the missing entries of a matrix given a sample of its 
entries. The key insight of Candes and Recht's approach is to formulate 
the matrix completion problem as a convex optimization problem, 
specifically, nuclear norm minimization.

The nuclear norm of a matrix, also known as the trace norm, is the sum 
of its singular values. Minimizing the nuclear norm subject to the 
constraint that the known entries remain fixed leads to a low-rank 
matrix that matches the observed data. This method is particularly 
effective when the underlying true matrix is of low rank or approximately 
low rank, which is common in many applications such as collaborative 
filtering, machine learning, and signal processing.

Algorithm Overview:
1. Formulation: The problem is formulated as minimizing the nuclear norm 
of the matrix subject to the constraint that its known entries match those 
of the original incomplete matrix.

2. Convex Optimization: The problem is solved using convex optimization 
techniques. The implementation may use solvers like CVXOPT, SCS, or 
others that support semidefinite programming.

3. Solution: The algorithm returns a completed matrix that has minimal 
nuclear norm among all matrices that agree with the observed entries.

Key Features:
- Robustness to noise: The approach is shown to be robust under certain 
conditions, even when the observed entries are subject to noise.
- Theoretical guarantees: Under certain conditions, the method guarantees 
exact recovery of the original matrix with high probability, given a 
sufficient number of observations.

Implementation Details:
- The `NuclearNormMinimization` class extends the `BaseImpute` abstract 
base class, providing a concrete implementation for matrix completion 
using convex optimization.
- Users can specify whether the solution must be symmetric, and adjust 
various optimization parameters through `training_options`.

Example Usage:
```python
from nuclear_norm import NuclearNormMinimization

# Initialize the imputer with desired options
imputer = NuclearNormMinimization(require_symmetric_solution=False)
completed_matrix = imputer.fit_transform(incomplete_matrix)
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cvxpy
from sklearn.utils import check_array

from bystro.imputation._base import BaseImpute


class NuclearNormMinimization(BaseImpute):
    def __init__(
        self,
        require_symmetric_solution: bool = False,
        training_options: Optional[Dict[str, Any]] = None,
        init_fill_method: str = "zero",
    ) -> None:
        super().__init__(
            fill_method=init_fill_method,
            training_options=training_options,
        )
        self.require_symmetric_solution = require_symmetric_solution

    def _constraints(
        self,
        X: cvxpy.Expression,
        missing_mask: np.ndarray,
        S: cvxpy.Variable,
        error_tolerance: float,
    ) -> list:
        """
        Constructs constraints for the nuclear norm minimization problem.

        Parameters:
        - X: cvxpy.Expression, the original data matrix with missing values.
        - missing_mask: np.ndarray, a boolean array where True indicates a missing value.
        - S: cvxpy.Variable, the variable representing the matrix to optimize.
        - error_tolerance: float, tolerance for reconstruction error.

        Returns:
        - list of cvxpy constraints.
        """
        ok_mask = ~missing_mask
        masked_X = cvxpy.multiply(ok_mask, X)
        masked_S = cvxpy.multiply(ok_mask, S)
        abs_diff = cvxpy.abs(masked_S - masked_X)
        close_to_data = abs_diff <= error_tolerance
        constraints = [close_to_data]
        if self.require_symmetric_solution:
            constraints.append(S == S.T)
        return constraints

    def _create_objective(
        self, m: int, n: int
    ) -> Tuple[cvxpy.Variable, cvxpy.Minimize]:
        """
        Creates the objective function for the optimization problem.

        Parameters:
        - m: int, number of rows in the matrix.
        - n: int, number of columns in the matrix.

        Returns:
        - A tuple containing the cvxpy.Variable representing the matrix S
          and the cvxpy.Minimize objective.
        """
        shape = (m, n)
        S = cvxpy.Variable(shape, name="S")
        norm = cvxpy.norm(S, "nuc")
        objective = cvxpy.Minimize(norm)
        return S, objective

    def _solve(self, X, missing_mask):
        """
        Solves the nuclear norm minimization problem to impute missing values.

        Parameters:
        - X: np.ndarray, the original data matrix with missing values.
        - missing_mask: np.ndarray, a boolean array where True indicates a missing value.

        Returns:
        - np.ndarray, the imputed data matrix.
        """
        to = self.training_options
        X = check_array(X, force_all_finite=False)

        m, n = X.shape
        S, objective = self._create_objective(m, n)
        constraints = self._constraints(
            X=X,
            missing_mask=missing_mask,
            S=S,
            error_tolerance=to["convergence_threshold"],
        )
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(
            max_iters=to["n_iterations"],
            use_indirect=False,
        )
        return S.value

    def _fill_training_options(self, training_options):
        """
        Fills in default training options where not provided.

        Parameters:
        - training_options: Dict[str, any], user-provided training options.

        Returns:
        - Dict[str, any], complete set of training options.
        """
        default_options = {
            "n_iterations": 100,
            "convergence_threshold": 0.001,
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
