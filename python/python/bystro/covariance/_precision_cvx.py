"""


"""
import numpy as np
import cvxpy as cp
import time
import cvxpy.atoms as at  # import log_det,norm_inf,norm_nuc,norm

from _base_precision import BasePrecision


class PrecisionClimePrimalCvx(BasePrecision):
    # http://www-stat.wharton.upenn.edu/~tcai/paper/Precision-Matrix.pdf

    def __init__(self, lambda_n=1.0):
        super().__init__()
        self.lambda_n = float(lambda_n)

    def __repr__(self):
        out_str = "PrecisionClimeCvxPrimal object\n"
        return out_str

    def fit(self, X, verbose=False):
        N, p = X.shape
        self.p = p

        I_p = np.eye(p)
        Sigma_n = np.dot(X.T, X) / N

        Omega = cp.Variable((p, p), PSD=True)
        cost = cp.norm(Omega, 1)
        inf_norm = at.norm_inf(Sigma_n @ Omega - I_p)

        constraint_list = [inf_norm <= self.lambda_n, Omega >> 0]
        prob = cp.Problem(cp.Minimize(cost), constraint_list)

        start_time = time.time()
        prob.solve(verbose=verbose)
        self.elapsed_time = time.time() - start_time

        self.optimal_value = prob.value
        self.precision = Omega.value
        self.fitted = True

        return self


class PrecisionClimeDualCvx(BasePrecision):
    # http://www-stat.wharton.upenn.edu/~tcai/paper/Precision-Matrix.pdf

    def __init__(self, mu_n):
        super().__init__()
        self.mu_n = float(mu_n)

    def __repr__(self):
        out_str = "PrecisionClimeCvxDual object\n"
        return out_str

    def fit(self, X, verbose=False):
        N, p = X.shape
        self.p = p

        I_p = np.eye(p)
        Sigma_n = np.dot(X.T, X) / N

        Omega = cp.Variable((p, p), PSD=True)
        cost = cp.norm(Omega, 1)
        inf_norm = at.norm_inf(Sigma_n @ Omega - I_p)

        cost_reg = cost + self.mu_n * inf_norm
        constraint_list = [Omega >> 0]

        prob = cp.Problem(cp.Minimize(cost_reg), constraint_list)

        start_time = time.time()
        prob.solve(verbose=verbose)
        self.elapsed_time = time.time() - start_time

        self.optimal_value = prob.value
        self.precision = Omega.value
        self.fitted = True


class PrecisionGlassoPrimalCvx(BasePrecision):
    # https://www.jmlr.org/papers/volume9/banerjee08a/banerjee08a.pdf

    def __init__(self, lambda_n=1.0):
        super().__init__()
        self.lambda_n = float(lambda_n)

    def __repr__(self):
        out_str = "PrecisionClimeCvxPrimal object\n"
        return out_str

    def fit(self, X, verbose=False):
        pass


class PrecisionL1PenalizedCvx(BasePrecision):
    # https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-5/issue-none/High-dimensional-covariance-estimation-by-minimizing-%e2%84%931-penalized-log-determinant/10.1214/11-EJS631.full

    def __init__(self, lambda_n=1.0):
        super().__init__()
        self.lambda_n = float(lambda_n)

    def fit(self, X, verbose=False):
        N, p = X.shape
        self.p = p

        np.eye(p)
        Sigma_n = np.dot(X.T, X) / N

        Omega = cp.Variable((p, p), PSD=True)
        loss_recon = sum(at.multiply(Omega, Sigma_n))
        loss_det = at.log_det(Omega)
        loss_off = cp.norm(Omega, 1) - cp.norm(at.diag(Omega), 1)
        cost = loss_recon - loss_det + self.lambda_n * loss_off

        constraint_list = [Omega >> 0]
        prob = cp.Problem(cp.Minimize(cost), constraint_list)

        start_time = time.time()
        prob.solve(verbose=verbose)
        self.elapsed_time = time.time() - start_time

        self.optimal_value = prob.value
        self.precision = Omega.value
        self.fitted = True
