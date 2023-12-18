"""

"""
import numpy as np
from numpy.typing import NDArray
import numpy.linalg as la
from ._base_precision import BasePrecision


class EmpiricalPrecision(BasePrecision):
    def __init__(self):
        super().__init__()

    def fit(self, X: NDArray):
        """
        This fits a precision matrix using samples X.

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data
        """
        self.N, self.p = X.shape
        XTX = np.dot(X.T, X)
        self.precision = la.inv(XTX / self.N)
