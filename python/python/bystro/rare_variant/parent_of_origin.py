import numpy as np
import numpy.linalg as la
from sklearn.covariance import ShrunkCovariance

def get_low_rank(A):
    U, s, Vt = la.svd(A)
    s[1:] = 0
    V = Vt.T
    V[:, 1:] = 0
    rank_1_approximation = np.dot(U * s, V.T)
    return rank_1_approximation


class BasePOE:
    def _test_inputs(self, X, y):
        if not isinstance(X, np.ndarray):
            raise ValueError("X is numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y is numpy array")
        if X.shape[0] != len(y):
            raise ValueError("X and y have different samples")


class POESingleSNP(BasePOE):
    def __init__(self, diagonalApproximation=False):
        self.diagonalApproximation = diagonalApproximation
        self.compute_pvalue = False
        self.n_permutations = 10000

    def fit(self, X, y):
        self._test_inputs(X, y)
        self.n_phenotypes = X.shape[1]

        X_homozygotes = X[y != 1]
        X_heterozygotes = X[y == 1]

        model_shrink = ShrunkCovariance()
        model_shrink.fit(X_homozygotes)
        Sigma_AA = model_shrink.covariance_
        model_shrink.fit(X_heterozygotes)
        Sigma_AB = model_shrink.covariance_

        B_est = Sigma_AB - Sigma_AA
        B_est_hat = get_low_rank(B_est)

        self.B_estimate = B_est_hat

        if self.diagonalApproximation:
            B_diag = np.maximum(np.diag(B_est_hat), 0)
            self.parent_effect_ = 4 * B_diag
        else:
            evals, evecs = la.eig((B_est_hat + B_est_hat.T) / 2)
            rev = np.abs(np.real(evals))
            idx_eval = np.where(rev == np.amax(rev))[0][0]
            self.parent_effect_ = np.real(evecs[:, idx_eval])
