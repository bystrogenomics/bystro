"""
This implements a variety of 

https://www.jmlr.org/papers/volume9/banerjee08a/banerjee08a.pdf

Objects
-------
CovarianceGlassoDualCvx(lambda_n=1)
    Sigma = arg max log det(Sigma) - trace(SSigma) - lambda||X||_1

Methods
-------
None
"""
import numpy as np
import cvxpy as cp
import time
import cvxpy.atoms as at

from ._base_covariance import BaseCovariance


class CovarianceGlassoPrimalCvx(BaseCovariance):
    def __init__(self, lambda_n):
        """ """
        super().__init__()
        self.lambda_n = float(lambda_n)

    def fit(self, X, verbose=True):
        """
        Fits the objective

        Sigma = arg max log det(Sigma) - trace(SSigma) - lambda||X||_1

        Parameters
        ----------
        X : np.array-like,(N,p)
            The data

        verbose : bool,default=True
            Whether to print default output of cvxpy

        Returns
        -------
        self : object_instance
            The object, allowing for model.fit(X).othermethod(params)
        """
        N, self.p = X.shape
        p = self.p

        S = np.dot(X.T, X) / N

        Sigma = cp.Variable((p, p), PSD=True)
        term1 = at.log_det(Sigma)
        term2 = cp.trace(Sigma @ S)
        term3 = at.norm.norm(Sigma, 1)
        cost = term1 - term2 - self.lambda_n * term3

        prob = cp.Problem(cp.Maximize(cost))

        start_time = time.time()
        prob.solve(verbose=verbose)
        self.elapsed_time = time.time() - start_time

        self.optimal_value = prob.value
        self.covariance = Sigma.value
        self.fitted = True

        return self
