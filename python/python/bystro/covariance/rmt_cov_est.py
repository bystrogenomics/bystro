import numpy as np
from pymanopt.manifolds import SymmetricPositiveDefinite  # type: ignore
from pymanopt import Problem  # type: ignore
from pymanopt.optimizers import SteepestDescent  # type: ignore
from pymanopt.function import numpy as pymanopt_function  # type: ignore
from scipy.special import spence  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def rmt_estim(
    S: np.ndarray, Cs: np.ndarray, n2: int, distance: str
) -> tuple[np.ndarray, float]:
    """
    Estimate the regularized covariance matrix based on the specified distance metric.

    Parameters
    ----------
    S : np.ndarray of shape (p, p)
        The input covariance matrix.

    Cs : np.ndarray of shape (p, p)
        The covariance matrix to compare against.

    n2 : int
        The number of samples.

    distance : str
        The distance metric to use for the estimation. Supported values include:
        ['Fisher', 'log', 't', 'log1st', 'KL', 'Battacharrya',
         'Inverse_Fisher', 'Inverse_log', 'Inverse_log1st',
         'Inverse_t', 'Inverse_KL', 'Inverse_Battacharrya'].

    Returns
    -------
    out : np.ndarray
        The estimated output based on the chosen distance metric.

    r : float
        The sign of the estimated output.
    """
    p = S.shape[0]

    if distance == "Fisher":
        F = Cs / S
        c2 = p / n2
        lambda_ = np.sort(np.diag(F))
        lambdat = np.concatenate(([0], lambda_))
        zeta = np.zeros(p)

        for i in range(p):
            kappa_p, kappa_m = lambdat[i + 1], lambdat[i]
            while abs(kappa_p - kappa_m) > 1e-6 * abs(lambdat[0] - lambdat[-1]):
                zeta_ = (kappa_p + kappa_m) / 2
                if (1 / n2) * np.sum(lambda_ / (lambda_ - zeta_)) < 1:
                    kappa_m = zeta_
                else:
                    kappa_p = zeta_
            zeta[i] = (kappa_p + kappa_m) / 2

        ker_vec = (2 / p) * (
            (lambda_ / lambda_[:, None])
            * np.log(abs(lambda_ / lambda_[:, None]))
            / (1 - lambda_ / lambda_[:, None])
        )
        ker_vec[np.isnan(ker_vec)] = 0
        ker = np.sum(ker_vec) - 2
        out = (
            (2 / p)
            * np.sum(
                (
                    (zeta / lambda_[:, None])
                    * np.log(abs(zeta / lambda_[:, None]))
                    / (1 - zeta / lambda_[:, None])
                )
            ).real
            - ker
            + (2 / p) * np.sum(np.log(lambda_))
            - (1 - c2)
            / c2
            * (
                np.log(1 - c2) ** 2
                + np.sum(np.log(lambda_) ** 2 - np.log(zeta) ** 2)
            )
            - 1
            / p
            * (
                2 * np.sum(spence(1 - (zeta[:, None] / lambda_)))
                - np.sum(np.log(lambda_) ** 2)
            )
        )
        r = np.sign(out)
        out = out**2

    elif distance == "log":
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_hatC2 = np.sort(np.linalg.eigvals(F))
        out = (
            1 / p * np.sum(np.log(lambda_hatC2))
            + (1 - c2) / c2 * np.log(1 - c2)
            + 1
        )
        r = np.sign(out)
        out = out**2

    elif distance == "t":
        F = np.linalg.solve(S, Cs)
        lambda_hatC2 = np.sort(np.linalg.eigvals(F))
        out1 = np.mean(lambda_hatC2)
        r = np.sign(out1)
        out = out1**2

    elif distance == "log1st":
        s = 1
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if 1 - m(kappa_) < 0:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa_0_hatC2 = (kappa_p + kappa_m) / 2
        out = (
            1 + s * kappa_0_hatC2 + np.log(abs(-s * kappa_0_hatC2))
        ) / c2 + 1 / p * np.sum(np.log(abs(1 - lambda_ / kappa_0_hatC2)))
        r = np.sign(out)
        out = out**2

    elif distance == "KL":
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_hatC2 = np.sort(np.linalg.eigvals(F))
        out = (
            -1
            / 2
            * (
                1 / p * np.sum(np.log(lambda_hatC2))
                + (1 - c2) / c2 * np.log(1 - c2)
                + 1
            )
            - 1 / 2
            + 1 / 2 * np.mean(lambda_hatC2)
        )
        r = np.sign(out)
        out = out**2

    elif distance == "Battacharrya":
        s = 1
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if 1 - m(kappa_) < 0:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa_0_hatC2 = (kappa_p + kappa_m) / 2
        out = (
            1
            / 2
            * (
                (1 + s * kappa_0_hatC2 + np.log(abs(-s * kappa_0_hatC2))) / c2
                + 1 / p * np.sum(np.log(abs(1 - lambda_ / kappa_0_hatC2)))
            )
            - 1
            / 4
            * (
                1 / p * np.sum(np.log(lambda_))
                + (1 - c2) / c2 * np.log(1 - c2)
                + 1
            )
            - 1 / 2 * np.log(2)
        )
        r = np.sign(out)
        out = out**2

    elif distance == "Inverse_Fisher":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        zeta = np.sort(
            np.linalg.eigvals(
                np.diag(lambda_)
                - (1 / n2) * np.sqrt(lambda_)[:, None] * np.sqrt(lambda_)
            )
        )
        ker_vec = (2 / p) * (
            (lambda_[:, None] / lambda_)
            * np.log(abs(lambda_[:, None] / lambda_))
            / (1 - lambda_[:, None] / lambda_)
        )
        ker_vec[np.isnan(ker_vec)] = 0
        ker = np.sum(ker_vec) - 2
        out = (
            (2 / p)
            * np.sum(
                (
                    (zeta[:, None] / lambda_)
                    * np.log(abs(zeta[:, None] / lambda_))
                    / (1 - zeta[:, None] / lambda_)
                )
            ).real
            - ker
            + (2 / p) * np.sum(np.log(lambda_))
            - (1 - c2)
            / c2
            * (
                np.log(1 - c2) ** 2
                - np.log(1) ** 2
                + np.sum(np.log(lambda_) ** 2 - np.log(zeta) ** 2)
            )
            - 1
            / p
            * (
                2 * np.sum(spence(1 - (zeta[:, None] / lambda_)))
                - np.sum(np.log((1 - 0) * lambda_) ** 2)
            )
        )
        r = np.sign(out)
        out = out**2

    elif distance == "Inverse_log":
        F = Cs @ S
        c2 = p / n2
        lambda_hatC2 = np.sort(np.linalg.eigvals(F).real)
        out = (
            -1 / p * np.sum(np.log(lambda_hatC2))
            - (1 - c2) / c2 * np.log(1 - c2)
            - 1
        )
        r = np.sign(out)
        out = out**2

    elif distance == "Inverse_log1st":
        s = 1
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa = np.zeros(p + 1)
        lambda1 = np.concatenate(([0], lambda_))
        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if m(kappa_) > 1 / s:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa[0] = (kappa_p + kappa_m) / 2
        for i in range(1, len(lambda1)):
            kappa_p, kappa_m = lambda1[i], lambda1[i - 1]
            while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
                kappa_ = (kappa_p + kappa_m) / 2
                if m(kappa_) > 1 / s:
                    kappa_p = kappa_
                else:
                    kappa_m = kappa_
            kappa[i] = (kappa_p + kappa_m) / 2
        out = (
            (1 / c2) * np.sum(lambda_ - kappa[1:])
            - 1 / p * np.sum(np.log(lambda_))
            + 1 / p * np.sum(np.log(lambda_ - kappa[0]))
            + (1 / c2 - 1) * np.sum(np.log(lambda_ / kappa[1:]))
            - 1
        )
        r = np.sign(out)
        out = out**2

    elif distance == "Inverse_t":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        out = 0
        for i in range(len(lambda_)):
            lambdac = np.delete(lambda_, i)
            out -= (
                1
                / p
                * (
                    (c2 / p) * np.sum(1 / (lambda_[i] - lambdac))
                    - (1 - c2) / lambda_[i]
                )
            )
        r = np.sign(out)
        out = out**2

    elif distance == "Inverse_Battacharrya":
        s = 1
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa = np.zeros(p + 1)
        lambda1 = np.concatenate(([0], lambda_))
        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if m(kappa_) > 1 / s:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa[0] = (kappa_p + kappa_m) / 2
        for i in range(1, len(lambda1)):
            kappa_p, kappa_m = lambda1[i], lambda1[i - 1]
            while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
                kappa_ = (kappa_p + kappa_m) / 2
                if m(kappa_) > 1 / s:
                    kappa_p = kappa_
                else:
                    kappa_m = kappa_
            kappa[i] = (kappa_p + kappa_m) / 2
        out = (
            1
            / 2
            * (
                (1 / c2) * np.sum(lambda_ - kappa[1:])
                - 1 / p * np.sum(np.log(lambda_))
                + 1 / p * np.sum(np.log(lambda_ - kappa[0]))
                + (1 / c2 - 1) * np.sum(np.log(lambda_ / kappa[1:]))
                - 1
            )
            - 1
            / 4
            * (
                -1 / p * np.sum(np.log(lambda_))
                - (1 - c2) / c2 * np.log(1 - c2)
                - 1
            )
            - 1 / 2 * np.log(2)
        )
        r = np.sign(out)
        out = out**2

    elif distance == "Inverse_KL":
        F = Cs @ S
        c2 = p / n2
        lambda_hatC2 = np.sort(np.linalg.eigvals(F).real)
        out_t = 0
        for i in range(len(lambda_hatC2)):
            lambdac = np.delete(lambda_hatC2, i)
            out_t -= (
                1
                / p
                * (
                    (c2 / p) * np.sum(1 / (lambda_hatC2[i] - lambdac))
                    - (1 - c2) / lambda_hatC2[i]
                )
            )
        out = (
            -1
            / 2
            * (
                -1 / p * np.sum(np.log(lambda_hatC2))
                - (1 - c2) / c2 * np.log(1 - c2)
                - 1
            )
            + 1 / 2 * out_t
            - 1 / 2
        )
        r = np.sign(out)
        out = out**2
    else:
        raise ValueError(
            f"Invalid distance type: {distance}. Please choose a valid distance metric."
        )

    return out, r


def rmt_estim_rgrad(
    S: np.ndarray, Cs: np.ndarray, n2: int, distance: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Riemannian gradient of the regularized covariance matrix estimation.

    Parameters
    ----------
    S : np.ndarray of shape (p, p)
        The input covariance matrix.

    Cs : np.ndarray of shape (p, p)
        The covariance matrix to compare against.

    n2 : int
        The number of samples.

    distance : str
        The distance metric to use for the estimation. Supported values include:
        ['Fisher', 'log', 't', 'log1st', 'KL', 'Battacharrya',
         'Inverse_Fisher', 'Inverse_log', 'Inverse_log1st',
         'Inverse_t', 'Inverse_KL', 'Inverse_Battacharrya'].

    Returns
    -------
    out_ : np.ndarray of shape (p, p)
        The computed Riemannian gradient.

    out_1 : np.ndarray of shape (p, p)
        The intermediate output matrix used in the computation.
    """
    p = S.shape[0]

    if distance == "Fisher":
        V = Cs / S
        c2 = p / n2
        lambda_ = np.sort(np.diag(V))
        lambdat = np.concatenate(([0], lambda_))
        zeta = np.zeros(p)

        for i in range(p):
            kappa_p, kappa_m = lambdat[i + 1], lambdat[i]
            while abs(kappa_p - kappa_m) > 1e-6 * abs(lambdat[0] - lambdat[-1]):
                zeta_ = (kappa_p + kappa_m) / 2
                if (1 / n2) * np.sum(lambda_ / (lambda_ - zeta_)) < 1:
                    kappa_m = zeta_
                else:
                    kappa_p = zeta_
            zeta[i] = (kappa_p + kappa_m) / 2

        M1 = np.divide(
            (2 / p) * np.log(lambda_[:, None]), (lambda_[:, None] - lambda_)
        )
        M2 = np.divide(
            (-2 / p) * np.log(zeta[:, None]), (lambda_[:, None] - zeta)
        )
        M3 = np.divide(-1, (lambda_[:, None] - lambda_))
        M4 = np.divide(1, (lambda_[:, None] - zeta))
        np.fill_diagonal(M1, 0)
        np.fill_diagonal(M3, 0)
        diag_out = (
            (2 / p)
            * np.log(lambda_)
            * (np.sum(M4, axis=1) + np.sum(M3, axis=1) - 1 / lambda_)
            + np.sum(M2, axis=1)
            + np.sum(M1, axis=1)
            - (2 - 2 * np.log(1 - c2)) / (p * lambda_)
        )
        vec = diag_out[np.argsort(np.diag(V))]
        my_out = Cs * np.diag(vec)
        out_1 = (my_out + my_out.T) / 2
        az, r = rmt_estim(S, Cs, n2, "Fisher")
        out_ = 2 * r * np.sqrt(az) * out_1
    elif distance == "log":
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        eigs, vecs = np.linalg.eig(F)
        eigs = np.sort(eigs.real)
        out_prim = np.diag(1 / eigs)
        out_p = -(1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = Cs @ vecs @ np.diag(vec) @ np.linalg.inv(vecs)
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "log")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "log1st":
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if 1 - m(kappa_) < 0:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa_0 = (kappa_p + kappa_m) / 2
        out_prim = np.diag(1 / (lambda_ - kappa_0))
        out_p = -(1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = Cs @ F @ np.diag(vec) @ np.linalg.inv(F)
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "log1st")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "t":
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        zeta = np.sort(
            np.linalg.eigvals(
                np.diag(lambda_)
                - (1 / n2) * np.sqrt(lambda_)[:, None] * np.sqrt(lambda_)
            )
        )

        def mp(z):
            return c2 * np.mean(1 / (lambda_ - z) ** 2) + (1 - c2) / z**2

        out_prim = np.zeros((p, p))
        for r in range(len(lambda_)):
            lambdac = np.delete(lambda_, r)
            ro = 0
            for g in range(len(zeta)):
                ro -= 1 / (mp(zeta[g]) * (zeta[g] - lambda_[r]) ** 2)
            out_prim[r, r] = ro + p / c2
        out_p = -(1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = Cs @ F @ np.diag(vec) @ np.linalg.inv(F)
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "t")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "KL":
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        zeta = np.sort(
            np.linalg.eigvals(
                np.diag(lambda_)
                - (1 / n2) * np.sqrt(lambda_)[:, None] * np.sqrt(lambda_)
            )
        )

        def mp(z):
            return c2 * np.mean(1 / (lambda_ - z) ** 2) + (1 - c2) / z**2

        out_pr = np.zeros((p, p))
        for r in range(len(lambda_)):
            lambdac = np.delete(lambda_, r)
            ro = 0
            for g in range(len(zeta)):
                ro -= 1 / (mp(zeta[g]) * (zeta[g] - lambda_[r]) ** 2)
            out_pr[r, r] = ro + p / c2
        out_prim = np.diag(-1 / 2 * (1 / lambda_) + 1 / 2 * out_pr.diagonal())
        out_p = -(1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = Cs @ F @ np.diag(vec) @ np.linalg.inv(F)
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "KL")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "Battacharrya":
        F = np.linalg.solve(S, Cs)
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if 1 - m(kappa_) < 0:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa_0 = (kappa_p + kappa_m) / 2
        out_prim = np.diag(
            1 / 2 * (1 / (lambda_ - kappa_0)) - 1 / 4 * (1 / lambda_)
        )
        out_p = -(1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = Cs @ F @ np.diag(vec) @ np.linalg.inv(F)
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "Battacharrya")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "Inverse_Fisher":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        zeta = np.sort(
            np.linalg.eigvals(
                np.diag(lambda_)
                - (1 / n2) * np.sqrt(lambda_)[:, None] * np.sqrt(lambda_)
            )
        )
        out_prim = np.zeros((p, p))
        for i in range(len(lambda_)):
            lambdac = np.delete(lambda_, i)
            conn3 = (
                -1 / lambda_[i]
                + np.sum(1 / (lambda_[i] - zeta))
                - np.sum(1 / (lambda_[i] - lambdac))
            )
            c = -np.log(1 - c2)
            out_prim[i, i] = (
                (-2 * conn3) * np.log(lambda_[i])
                + (2 + 2 * c) / lambda_[i]
                + 2 * np.sum(1 / (lambda_[i] - zeta) * np.log(zeta))
                - 2 * np.sum(1 / (lambda_[i] - lambdac) * np.log(lambdac))
            )
        out_p = -(1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = -S @ F @ np.diag(vec) @ np.linalg.inv(F) @ Cs @ S
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "Inverse_Fisher")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "Inverse_log":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        out_prim = np.diag(1 / lambda_)
        out_p = (1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = -S @ F @ np.diag(vec) @ np.linalg.inv(F) @ Cs @ S
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "Inverse_log")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "Inverse_log1st":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if 1 - m(kappa_) < 0:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa_0 = (kappa_p + kappa_m) / 2
        out_prim = np.diag(-1 / (lambda_ - kappa_0) + 1 / lambda_)
        out_p = (1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = -S @ F @ np.diag(vec) @ np.linalg.inv(F) @ Cs @ S
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "Inverse_log1st")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "Inverse_t":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        out_prim = np.diag((1 - c2) / lambda_**2)
        out_p = (1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = -S @ F @ np.diag(vec) @ np.linalg.inv(F) @ Cs @ S
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "Inverse_t")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "Inverse_KL":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)
        out_prim = np.diag(
            -1 / 2 * (1 / lambda_) + 1 / 2 * (1 - c2) / lambda_**2
        )
        out_p = (1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = -S @ F @ np.diag(vec) @ np.linalg.inv(F) @ Cs @ S
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "Inverse_KL")
        out_ = 2 * r * np.sqrt(a) * out_1

    elif distance == "Inverse_Battacharrya":
        F = Cs @ S
        c2 = p / n2
        lambda_ = np.sort(np.linalg.eigvals(F).real)

        def m(z):
            return c2 * np.mean(1 / (lambda_ - z)) - (1 - c2) / z

        kappa_p, kappa_m = 0, -10
        while abs(kappa_p - kappa_m) > 1e-7 * abs(lambda_[-1] - lambda_[0]):
            kappa_ = (kappa_p + kappa_m) / 2
            if 1 - m(kappa_) < 0:
                kappa_p = kappa_
            else:
                kappa_m = kappa_
        kappa_0 = (kappa_p + kappa_m) / 2
        out_prim = np.diag(
            1 / 2 * (-1 / (lambda_ - kappa_0) + 1 / lambda_)
            - 1 / 4 * (1 / lambda_)
        )
        out_p = (1 / p) * out_prim
        diag_out = np.diag(out_p)
        vec = diag_out[np.argsort(np.diag(F))]
        my_out = -S @ F @ np.diag(vec) @ np.linalg.inv(F) @ Cs @ S
        out_1 = (my_out + my_out.T) / 2
        a, r = rmt_estim(S, Cs, n2, "Inverse_Battacharrya")
        out_ = 2 * r * np.sqrt(a) * out_1
    else:
        raise ValueError(
            f"Invalid distance type: {distance}. Please choose a valid distance metric."
        )

    return out_, out_1


def rmtest(
    x: np.ndarray,
    C0: np.ndarray,
    C: np.ndarray,
    check_gradient: bool,
    plot_cost: bool,
    distance: str,
) -> tuple[np.ndarray, dict]:
    """
    Perform Riemannian manifold testing to estimate the covariance matrix.

    Parameters
    ----------
    x : np.ndarray of shape (p, n)
        The input data matrix.

    C0 : np.ndarray of shape (p, p)
        The initial guess for the covariance matrix.

    C : np.ndarray of shape (p, p)
        The true covariance matrix.

    check_gradient : bool
        Whether to check the gradient numerically.

    plot_cost : bool
        Whether to plot the cost function during optimization.

    distance : str
        The distance metric to use for the estimation. Supported values include:
        ['Fisher', 'log', 't', 'log1st', 'KL', 'Battacharrya',
         'Inverse_Fisher', 'Inverse_log', 'Inverse_log1st',
         'Inverse_t', 'Inverse_KL', 'Inverse_Battacharrya'].

    Returns
    -------
    C_est : np.ndarray of shape (p, p)
        The estimated covariance matrix after optimization.

    log : dict
        The log containing optimization details.
    """
    p, n = x.shape

    # Define the distance function
    if distance in ["log", "Inverse_log"]:

        def f(z):
            return np.log(z)

    elif distance in ["Fisher", "Inverse_Fisher"]:

        def f(z):
            return np.log(z) ** 2

    elif distance in ["log1st", "Inverse_log1st"]:
        s = 1

        def f(z):
            return np.log(1 + s * z)

    elif distance in ["t", "Inverse_t"]:

        def f(z):
            return z

    elif distance in ["Battacharrya", "Inverse_Battacharrya"]:

        def f(z):
            return (
                -(1 / 4) * np.log(z) + 1 / 2 * np.log(1 + z) - 1 / 2 * np.log(2)
            )

    elif distance in ["KL", "Inverse_KL"]:

        def f(z):
            return -(1 / 2) * np.log(z) + 1 / 2 * z - 1 / 2

    else:
        raise ValueError("Unknown distance type")

    # Compute the sample covariance matrix
    U, Cs = np.linalg.eigh(x @ x.T / n)

    # Define the manifold
    man = SymmetricPositiveDefinite(p)

    @pymanopt_function(man)
    def cost_function(S):
        return rmt_estim(S, Cs, n, distance)[0]

    @pymanopt_function(man)
    def rgrad_function(S):
        return rmt_estim_rgrad(S, Cs, n, distance)[1]

    # Define the problem
    problem = Problem(
        manifold=man, cost=cost_function, riemannian_gradient=rgrad_function
    )

    # Check gradient numerically if required
    if check_gradient:
        problem.check_gradient()

    # Optimization options
    solver = SteepestDescent(verbosity=0)

    # Ensure initial_point is on the manifold and has the correct shape
    C0 = np.array(C0)  # Ensure C0 is a numpy array
    if C0.shape != (p, p):
        raise ValueError(
            f"Initial point shape mismatch: expected {(p, p)}, got {C0.shape}"
        )

    L_est = solver.run(problem, initial_point=C0).point

    # Reconstruct the estimated covariance matrix
    C_est = U @ L_est @ U.T

    # Plot cost function if required
    if plot_cost:
        info = solver._log  # noqa: SLF001
        matrix = np.array(
            [info[i]["x"].reshape((p, p)) for i in range(len(info))]
        )

        if distance in ["Fisher", "log", "log1st", "t", "Battacharrya", "KL"]:

            def metric(D):
                return np.mean(f(np.linalg.eigvalsh(np.linalg.inv(D) @ C)))

        else:

            def metric(D):
                return np.mean(
                    f(np.linalg.eigvalsh(np.linalg.inv(D) @ np.linalg.inv(C)))
                )

        matrix_int = np.array(
            [metric(matrix[i]) for i in range(matrix.shape[0])]
        )

        plt.figure(2)
        plt.plot(
            matrix_int, "r*-", linewidth=2, markersize=4, label="real distance"
        )
        plt.plot(
            np.sqrt([info[i]["cost"] for i in range(len(info))]),
            "go-",
            linewidth=2,
            markersize=4,
            label="cost function",
        )
        plt.legend()
        plt.show()
    return C_est, solver._log["optimizer_parameters"]  # noqa: SLF001
