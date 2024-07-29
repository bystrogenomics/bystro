import numpy as np
from scipy.linalg import eigh  # type: ignore
from scipy.stats import norm, chi2  # type: ignore
from scipy.sparse.linalg import eigs  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore
import logging
from typing import Optional, List, Union, Dict, Any

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def two_way_sampling(
    X: np.ndarray, Y: np.ndarray, n: int, seed: int = 2021
) -> dict:
    """
    Perform two-way sampling on matrices X and Y.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : int
        Sample size.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    sampled_matrices : dict
        Dictionary containing sampled matrices Xs, Ys, Zs.
    """
    rng = np.random.default_rng(seed)
    n1, n2 = X.shape[0], Y.shape[0]
    if n > min(max(n1, n2) / 2, n1, n2):
        raise ValueError("Invalid sample sizes!")

    if n1 <= n2:
        xs_idx = rng.choice(n1, n, replace=False)
        ys_idx = rng.choice(n2, 2 * n, replace=False)
        zs_idx = ys_idx[n:]
        ys_idx = ys_idx[:n]
        Xs = X[xs_idx, :]
        Ys = Y[ys_idx, :]
        Zs = Y[zs_idx, :]
    else:
        xs_idx = rng.choice(n1, 2 * n, replace=False)
        ys_idx = rng.choice(n2, n, replace=False)
        zs_idx = xs_idx[n:]
        xs_idx = xs_idx[:n]
        Xs = X[xs_idx, :]
        Ys = Y[ys_idx, :]
        Zs = X[zs_idx, :]

    return {"Xs": Xs, "Ys": Ys, "Zs": Zs}


def cov_eigs(X: np.ndarray) -> np.ndarray:
    """
    Find eigenvalues of the sample covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.

    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues of the sample covariance matrix.
    """
    n, p = X.shape
    X1 = X - np.mean(X, axis=0)
    return eigh(np.dot(X1, X1.T) / np.sqrt(n * p), eigvals_only=True)


def k_func(x: float) -> float:
    """
    Kernel function K.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    k_value : float
        Output of the kernel function.
    """
    if abs(x) >= 1.05:
        return 0
    if abs(x) <= 1:
        return 1
    return np.exp(1 / 0.05**2 - 1 / (0.05**2 - (abs(x) - 1) ** 2))


def t_func(lambda_: np.ndarray, gamma: float, eta0: float) -> float:
    """
    Compute T value.

    Parameters
    ----------
    lambda_ : np.ndarray
        Array of eigenvalues.
    gamma : float
        Gamma value.
    eta0 : float
        Eta value.

    Returns
    -------
    t_value : float
        T value.
    """
    return sum(
        (lambda_i - gamma) / eta0 * k_func((lambda_i - gamma) / eta0)
        for lambda_i in lambda_
    )


def check_efficient(
    gamma: float,
    lambda1: np.ndarray,
    lambda2: np.ndarray,
    epsilon: float = 0.05,
) -> bool:
    """
    Check if the splitting is efficient.
    Parameters
    ----------
    gamma : float
        Gamma value.
    lambda1 : np.ndarray
        First array of eigenvalues.
    lambda2 : np.ndarray
        Second array of eigenvalues.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    Returns
    -------
    is_efficient : bool
        True if efficient, False otherwise.
    """
    range1 = abs(lambda1[0] - lambda1[-1])
    range2 = abs(lambda2[0] - lambda2[-1])
    return not (
        max(abs(gamma - lambda1[0]), abs(gamma - lambda1[-1]))
        > range1 - epsilon
        or max(abs(gamma - lambda2[0]), abs(gamma - lambda2[-1]))
        > range2 - epsilon
    )


def two_sample_test_(
    X: np.ndarray,
    Y: np.ndarray,
    n: int,
    const: float = 0.5,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    thres: Optional[float] = None,
    mode: str = "test",
    seed: int = 2021,
) -> Union[dict, float]:
    """
    Perform a two-sample test.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : int
        Sample size.
    const : float, optional
        Constant value, by default 0.5.
    alpha : float, optional
        Significance level, by default 0.05.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    thres : Optional[float], optional
        Threshold value, by default None.
    mode : str, optional
        Mode of the test, by default "test".
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    test_result : dict
        Result of the two-sample test.
    """
    sample_list = two_way_sampling(X, Y, n, seed=seed)
    Xs, Ys, Zs = sample_list["Xs"], sample_list["Ys"], sample_list["Zs"]

    eig_xs = cov_eigs(Xs)
    eig_ys = cov_eigs(Ys)
    eig_zs = cov_eigs(Zs)
    gamma = np.median(eig_zs)
    if not check_efficient(gamma, eig_xs, eig_ys, epsilon):
        return {"efficient": False, "c": 1}

    eta0 = np.std(eig_zs, ddof=1) * const

    Tx = t_func(eig_xs, gamma, eta0)
    Ty = t_func(eig_ys, gamma, eta0)

    if mode != "test":
        return abs(Tx - Ty)

    if thres is None:
        logger.warning("Threshold thres not specified, setting %f", 2.6)
        thres = 2.6

    if abs(Tx - Ty) > (
        thres / norm.ppf(1 - 0.05 / 2) * norm.ppf(1 - alpha / 2)
    ):
        return {
            "efficient": True,
            "c": 1,
            "statistic": (Tx - Ty) / (thres / norm.ppf(1 - 0.05 / 2)),
        }
    return {
        "efficient": True,
        "c": 0,
        "statistic": (Tx - Ty) / (thres / norm.ppf(1 - 0.05 / 2)),
    }


def two_sample_test(
    X: np.ndarray,
    Y: np.ndarray,
    n: Optional[int] = None,
    k: int = 100,
    const: Optional[float] = None,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    thres: Optional[float] = None,
    calib: bool = False,
    seed: int = 2021,
) -> dict:
    """
    Perform a two-sample test multiple times.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : Optional[int], optional
        Sample size, by default None.
    k : int, optional
        Number of iterations, by default 100.
    const : Optional[float], optional
        Constant value, by default None.
    alpha : float, optional
        Significance level, by default 0.05.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    thres : Optional[float], optional
        Threshold value, by default None.
    calib : bool, optional
        Calibration flag, by default False.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    test_result : dict
        Result of the two-sample test.
    """

    if const is not None and (
        not isinstance(const, float) or not (0.1 <= const <= 10)
    ):
        raise ValueError("const must be a float between 0.1 and 10.")
    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError("alpha must be a float between 0 and 1.")
    if not isinstance(epsilon, float) or not (0 < epsilon < 1):
        raise ValueError("epsilon must be a float between 0 and 1.")
    if thres is not None and (not isinstance(thres, float) or thres <= 0):
        raise ValueError("thres must be a float greater than 0.")

    rng = np.random.default_rng(seed)
    reject, df, statistic = 0, 0, 0

    if n is None:
        n1, n2 = X.shape[0], Y.shape[0]
        n = int(min(max(n1, n2) / 2, n1, n2)) - 5

    if thres is None:
        if not calib:
            logger.warning("Threshold thres not specified, setting %f", 2.6)
            thres = 2.6
        else:
            thres = calibration(
                n1=X.shape[0],
                n2=Y.shape[0],
                p=X.shape[1],
                n=n,
                alpha=alpha,
                const=0.5,
                iterations=100,
                seed=seed,
            )

    if const is None:
        const = c_tuning(
            X, Y, n, alpha=alpha, epsilon=epsilon, thres=thres, seed=seed
        )["c"]

    for _ in range(k):
        result = two_sample_test_(
            X,
            Y,
            n,
            alpha=alpha,
            const=const,
            epsilon=epsilon,
            thres=thres,
            seed=rng.integers(int(1e6)),
        )
        if isinstance(result, dict) and result["efficient"]:
            df += 1
            statistic += result["statistic"] ** 2
            if result["c"] == 1:
                reject += 1

    if df < 10:
        pvalue = 0
        statistic = 10000
    else:
        pvalue = 1 - chi2.cdf(statistic, df)

    return {
        "decision": int(reject > rng.binomial(k, alpha)),
        "statistic": statistic,
        "pvalue": pvalue,
        "df": df,
        "reject": reject,
    }


def calibration(
    n1: int,
    n2: int,
    p: int,
    n: int,
    alpha: float = 0.05,
    const: float = 0.5,
    iterations: int = 100,
    K: int = 100,
    seed: int = 2021,
) -> float:
    """
    Perform calibration to find threshold.

    Parameters
    ----------
    n1 : int
        Size of the first sample.
    n2 : int
        Size of the second sample.
    p : int
        Number of columns.
    n : int
        Sample size.
    alpha : float, optional
        Significance level, by default 0.05.
    const : float, optional
        Constant value, by default 0.5.
    iterations : int, optional
        Number of iterations, by default 100.
    K : int, optional
        Number of sub-iterations, by default 100.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(iterations):
        X = rng.normal(0, 1, (n1, p))
        Y = rng.normal(0, 1, (n2, p))
        for _ in range(K):
            value = two_sample_test_(
                X,
                Y,
                n,
                alpha=alpha,
                const=const,
                mode="calib",
                seed=rng.integers(int(1e6)),
            )
            if isinstance(value, float):
                values.append(value)
    return float(np.quantile(values, 1 - alpha))


def c_tuning(
    X: np.ndarray,
    Y: np.ndarray,
    n: int,
    thres: Optional[float] = None,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    K: int = 500,
    seed: int = 2021,
) -> dict:
    """
    Tune the constant C.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    n : int
        Sample size.
    thres : Optional[float], optional
        Threshold value, by default None.
    alpha : float, optional
        Significance level, by default 0.05.
    epsilon : float, optional
        Tolerance value, by default 0.05.
    K : int, optional
        Number of iterations, by default 500.
    seed : int, optional
        Random seed, by default 2021.

    Returns
    -------
    tuning_result : dict
        Tuned constant C and rates.
    """
    rng = np.random.default_rng(seed)
    Cs = np.arange(1, 31) / 10
    rates = []

    if thres is None:
        logger.warning("Threshold thres not specified, setting %f", 2.6)
        thres = 2.6

    for c in Cs:
        all_ = 0
        rej = 0
        for _ in range(K):
            result = two_sample_test_(
                X,
                Y,
                n,
                alpha=alpha,
                const=c,
                epsilon=epsilon,
                thres=thres,
                seed=rng.integers(int(1e6)),
            )
            if isinstance(result, dict) and result["efficient"]:
                all_ += 1
                rej += result["c"]
        rates.append(rej / all_ if all_ > 50 else 1)

    stable = find_stable(rates)
    return {"c": Cs[stable], "rates": rates}


def find_stable(xs: List[float]) -> int:
    """
    Find the stable point in a list of rates.

    Parameters
    ----------
    xs : list
        List of rates.

    Returns
    -------
    stable_index : int
        Index of the stable point.
    """
    roll_average = np.convolve(xs, np.ones(3) / 3, mode="valid")
    vars_ = [np.var(roll_average[:i]) for i in range(2, len(roll_average))]
    for i in range(len(vars_) - 1):
        if vars_[i + 1] < vars_[i] and roll_average[i] > max(roll_average) / 5:
            break
    return i + 2


def movevar(xs: List[float]) -> List[float]:
    """
    Compute the moving variance of a list.

    Parameters
    ----------
    xs : list
        List of values.

    Returns
    -------
    variances : list
        List of variances.
    """
    n = len(xs)
    vars_ = []
    for i in range(2, n + 1):
        vars_.append(float(np.var(xs[:i])))
    return vars_


#########################
#########################
##                     ##
##    Other methods    ##
##                     ##
#########################
#########################


def clx2013(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Perform the CLX2013 test.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.

    Returns
    -------
    test_result : dict
        Test statistic and p-value.
    """
    n1, p = X.shape
    n2 = Y.shape[0]

    W1 = X - np.mean(X, axis=0)
    W2 = Y - np.mean(Y, axis=0)

    S1 = np.dot(W1.T, W1) / n1
    S2 = np.dot(W2.T, W2) / n2

    Theta1 = np.zeros((p, p))
    Theta2 = np.zeros((p, p))

    for i in range(n1):
        Theta1 += (1 / n1) * (np.outer(W1[i, :], W1[i, :]) - S1) ** 2
    for i in range(n2):
        Theta2 += (1 / n2) * (np.outer(W2[i, :], W2[i, :]) - S2) ** 2

    W = (S1 - S2) / np.sqrt(Theta1 / n1 + Theta2 / n2)
    M = W**2
    M_n = np.max(M)

    TSvalue = M_n - 4 * np.log(p) + np.log(np.log(p))
    pvalue = 1 - np.exp(-1 / np.sqrt(8 * np.pi) * np.exp(-TSvalue / 2))

    return {"TSvalue": TSvalue, "pvalue": pvalue}


def sy2010(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Perform the SY2010 test.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.

    Returns
    -------
    test_result : dict
        Test statistic and p-value.
    """
    n1 = X.shape[0] - 1
    n2 = Y.shape[0] - 1
    m = X.shape[1]
    n = n1 + n2

    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    if m <= n:
        V1 = np.dot(X.T, X)
        V2 = np.dot(Y.T, Y)
    else:
        V1 = np.dot(X, X.T)
        V2 = np.dot(Y, Y.T)

    V = np.dot(X.T, X) + np.dot(Y.T, Y)

    a11 = np.sum(np.diag(V1)) / (m * n1)
    a12 = np.sum(np.diag(V2)) / (m * n2)

    a21 = (np.linalg.norm(V1, "fro") ** 2 - np.sum(np.diag(V1)) ** 2 / n1) / (
        m * (n1 - 1) * (n1 + 2)
    )
    a22 = (np.linalg.norm(V2, "fro") ** 2 - np.sum(np.diag(V2)) ** 2 / n2) / (
        m * (n2 - 1) * (n2 + 2)
    )

    r = min(m, n1 + n2 + 2)
    eigV = eigs(V, k=r, return_eigenvectors=False).real
    a1 = np.sum(eigV) / (n * m)
    a2 = (np.sum(eigV**2) - np.sum(np.diag(V)) ** 2 / n) / (
        m * (n - 1) * (n + 2)
    )
    a3 = (
        np.sum(eigV**3) / m
        - 3 * n * (n + 1) * m * a2 * a1
        - n * m**2 * a1**3
    ) / (n * (n**2 + 3 * n + 4))
    c0 = n * (n**3 + 6 * n**2 + 21 * n + 18)
    c1 = 2 * n * (2 * n**2 + 6 * n + 9)
    c2 = 2 * n * (3 * n + 2)
    c3 = n * (2 * n**2 + 5 * n + 7)
    a4 = (
        np.sum(eigV**4) / m
        - m * c1 * a1
        - m**2 * c2 * a1**2 * a2
        - m * c3 * a2**2
        - n * m**3 * a1**4
    ) / c0

    gamma1 = a21 / a11**2
    gamma2 = a22 / a12**2
    xi1_2 = (
        4
        / n1**2
        * (
            a2**2 / a1**4
            + 2
            * n1
            / m
            * (a2**3 / a1**6 - 2 * a2 * a3 / a1**5 + a4 / a1**4)
        )
    )
    xi2_2 = (
        4
        / n2**2
        * (
            a2**2 / a1**4
            + 2
            * n2
            / m
            * (a2**3 / a1**6 - 2 * a2 * a3 / a1**5 + a4 / a1**4)
        )
    )

    Q2 = (gamma1 - gamma2) ** 2 / (xi1_2 + xi2_2)
    pvalue = 1 - chi2.cdf(Q2, df=1)

    return {"Q2": Q2, "pvalue": pvalue}


def hc2018(
    X: np.ndarray, Y: np.ndarray, N: Optional[int] = None, alpha: float = 0.05
) -> dict:
    """
    Perform the HC2018 test.

    Parameters
    ----------
    X : np.ndarray
        First matrix.
    Y : np.ndarray
        Second matrix.
    N : Optional[int], optional
        Parameter for the test, by default None.
    alpha : float, optional
        Significance level, by default 0.05.

    Returns
    -------
    test_result : dict
        Test result.
    """
    if N is None:
        N = int(np.floor(X.shape[1] ** 0.7))

    def double_sum(X1, X2):
        result = np.sum(X1, axis=0) * np.sum(X2, axis=0) - np.sum(
            X1 * X2, axis=0
        )
        return result

    def triple_sum(X1, X2, X3):
        result = (
            double_sum(X1, X2) * np.sum(X3, axis=0)
            - double_sum(X1 * X3, X2)
            - double_sum(X1, X2 * X3)
        )
        return result

    def quad_sum(X1, X2, X3, X4):
        result = (
            triple_sum(X1, X2, X3) * np.sum(X4, axis=0)
            - triple_sum(X1 * X4, X2, X3)
            - triple_sum(X1, X2 * X4, X3)
            - triple_sum(X1, X2, X3 * X4)
        )
        return result

    def di(X, q):
        n, p = X.shape
        X1 = X[:, : p - q]
        X2 = X[:, q:]
        D_1 = np.sum(double_sum(X1 * X2, X1 * X2))
        D_2 = np.sum(triple_sum(X1, X2, X1 * X2))
        D_3 = np.sum(quad_sum(X1, X2, X1, X2))
        result = (
            1 / (n * (n - 1)) * D_1
            - 2 / (n * (n - 1) * (n - 1)) * D_2
            + 1 / (n * (n - 1) * (n - 2) * (n - 3)) * D_3
        )
        return result

    def dc(X1, X2, q):
        n1, p = X1.shape
        n2 = X2.shape[0]
        X11 = X1[:, : p - q]
        X12 = X1[:, q:]
        X21 = X2[:, : p - q]
        X22 = X2[:, q:]
        Dc_1 = np.sum(np.sum(X11 * X12, axis=0) * np.sum(X21 * X22, axis=0))
        Dc_2 = np.sum(double_sum(X11, X12) * np.sum(X21 * X22, axis=0))
        Dc_3 = np.sum(np.sum(X11 * X12, axis=0) * double_sum(X21, X22))
        Dc_4 = np.sum(double_sum(X11, X12) * double_sum(X21, X22))
        result = (
            Dc_1 / (n1 * n2)
            - Dc_2 / (n1 * (n1 - 1) * n2)
            - Dc_3 / (n1 * n2 * (n2 - 1))
            + Dc_4 / (n1 * (n1 - 1) * n2 * (n2 - 1))
        )
        return result

    def sq(X1, X2, q):
        result = di(X1, q) + di(X2, q) - 2 * dc(X1, X2, q)
        return result

    def ri(X, q):
        n, p = X.shape

        X = X - np.mean(X, axis=0)

        X1 = X[:, : p - q]

        X2 = X[:, q:]

        Y = X1 * X2

        Y = Y - np.sum(Y, axis=0) / (n - 1)

        YYt2 = np.dot(Y, Y.T) ** 2

        result = (np.sum(YYt2) - np.sum(np.diag(YYt2))) / (n * (n - 1))

        return result

    def rc(X1, X2, q):
        n1, p = X1.shape
        X1 = X1 - np.mean(X1, axis=0)
        X11 = X1[:, : p - q]
        X12 = X1[:, q:]
        Y1 = X11 * X12
        Y1 = Y1 - np.sum(Y1, axis=0) / (n1 - 1)

        n2 = X2.shape[0]
        X2 = X2 - np.mean(X2, axis=0)
        X21 = X2[:, : p - q]
        X22 = X2[:, q:]
        Y2 = X21 * X22
        Y2 = Y2 - np.sum(Y2, axis=0) / (n2 - 1)
        result = np.sum((np.dot(Y1, Y2.T)) ** 2) / (n1 * n2)

        return result

    def v2(X1, X2, q):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        result = (
            ri(X1, q) * 2 / (n1 * (n1 - 1))
            + ri(X2, q) * 2 / (n2 * (n2 - 1))
            + rc(X1, X2, q) * 4 / (n1 * n2)
        )
        return result

    def one_super(X1, X2, q):
        chi = sq(X1, X2, q) ** 2 / v2(X1, X2, q)
        result = chi2.sf(chi, 1)
        return result

    pvalues = [one_super(X, Y, i) for i in range(N + 1)]
    test = adaptive_sts(pvalues, alpha=alpha, lambda_=0.5, silent=True)

    return {"reject": np.sum(test["rejected"]), "pvalues": pvalues, "N": N}


def storey_pi0_est(pValues: List[float], lambda_: float = 0.5) -> dict:
    """
    Estimate the proportion of true null hypotheses.

    Parameters
    ----------
    pValues : list
        List of p-values.
    lambda_ : float, optional
        Lambda parameter, by default 0.5.

    Returns
    -------
    pi0_estimate : dict
        Estimated pi0.
    """
    pi0 = np.mean(np.array(pValues) > lambda_) / (1 - lambda_)
    pi0 = np.min((float(pi0), 1.0))
    return {"pi0": pi0}


def print_rejected(
    rejected: List[bool], pValues: List[float], adjPValues: List[float]
) -> None:
    """
    Print the rejected hypotheses.

    Parameters
    ----------
    rejected : list
        List of rejected hypotheses.
    pValues : list
        List of p-values.
    adjPValues : list
        List of adjusted p-values.
    """
    for i in range(len(pValues)):
        if rejected[i]:
            print(
                f"pValue: {pValues[i]}, adjPValue: {adjPValues[i]}, Rejected: {rejected[i]}"
            )


def adaptive_sts(
    pValues: List[float],
    alpha: float,
    lambda_: float = 0.5,
    silent: bool = False,
) -> dict:
    """
    Perform the adaptive step-up procedure.

    Parameters
    ----------
    pValues : list
        List of p-values.
    alpha : float
        Significance level.
    lambda_ : float, optional
        Lambda parameter, by default 0.5.
    silent : bool, optional
        Silent flag, by default False.

    Returns
    -------
    sts_result : dict
        Result of the adaptive step-up procedure.
    """
    m = len(pValues)
    _, adjP, _, _ = multipletests(pValues, alpha=alpha, method="fdr_bh")

    pi0 = storey_pi0_est(pValues, lambda_)["pi0"]
    criticalValues = [(i * alpha) / (m * pi0) for i in range(1, m + 1)]
    adjPValues = adjP * min(pi0, 1.0)
    rejected = adjPValues <= alpha

    if not silent:
        print(
            "\n\n\t\tStorey-Taylor-Siegmund (2004) adaptive step-up procedure\n\n"
        )
        print_rejected(rejected, pValues, adjPValues)

    return {
        "adjPValues": adjPValues,
        "criticalValues": criticalValues,
        "rejected": rejected,
        "pi0": pi0,
        "errorControl": {"type": "FDR", "alpha": alpha},
    }


######################
######################
##                  ##
##      HDTest      ##
##                  ##
######################
######################


def hd2017(
    X: np.ndarray,
    Y: np.ndarray,
    J: int = 2500,
    seed: int = 2021,
    dname: str = "X and Y",
) -> Dict[str, Any]:
    """
    Perform the Two-Sample HD test for the equality of two covariance matrices from
    'Chang, J., Zhou, W., Zhou, W.-X., and Wang, L. (2016). Comparing large covariance matrices
    under weak conditions on the dependence structure and its application to gene clustering.'

    Parameters
    ----------
    X : np.ndarray
        The n1 x p data matrix for sample 1.
    Y : np.ndarray
        The n2 x p data matrix for sample 2.
    J : int, optional
        The number of permutations, by default 2500.
    seed : int, optional
        The random seed for reproducibility, by default 2021.
    dname : str, optional
        The name of the data, by default "X and Y".

    Returns
    -------
    hd_res : dict
        A dictionary containing the test statistics, p-value,
        alternative hypothesis, method, and data names.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    n1, p = X.shape
    n2 = Y.shape[0]

    if Y.shape[1] != p:
        raise ValueError("Different dimensions of X and Y.")

    scalev = np.tile(np.concatenate([np.ones(n1) / n1, np.ones(n2) / n2]), J)

    Sx = np.cov(X, rowvar=False) * (n1 - 1) / n1
    Sy = np.cov(Y, rowvar=False) * (n2 - 1) / n2

    xa = X - np.mean(X, axis=0)
    ya = Y - np.mean(Y, axis=0)

    vx = ((xa**2).T @ (xa**2)) / n1 - 2 / n1 * ((xa.T @ xa) * Sx) + Sx**2
    vy = ((ya**2).T @ (ya**2)) / n2 - 2 / n2 * ((ya.T @ ya) * Sy) + Sy**2

    with np.errstate(invalid="ignore"):
        deno = np.sqrt(vx / n1 + vy / n2)
    numo = np.abs(Sx - Sy)
    Tnm = np.max(numo / deno)

    xat = xa.T / n1
    yat = ya.T / n2
    ts = np.zeros(J)

    rng = np.random.default_rng(seed)
    for j in range(J):
        g = rng.standard_normal(n1 + n2)
        scalev = np.concatenate([np.ones(n1) / n1, np.ones(n2) / n2])
        g *= scalev
        atmp = np.sum(g[:n1])
        btmp = np.sum(g[n1:])

        ts1 = ((xa * g[:n1][:, np.newaxis]) - (xat.T * atmp)).T @ xa
        ts2 = ((ya * g[n1:][:, np.newaxis]) - (yat.T * btmp)).T @ ya

        ts[j] = np.max(np.abs(ts1 - ts2) / deno)

    hd_res = {
        "statistics": Tnm,
        "p.value": np.mean(ts >= Tnm),
        "alternative": "two.sided",
        "method": "Two-Sample HD test",
        "data.name": dname,
    }

    return hd_res


def schott2007(
    X: np.ndarray, Y: np.ndarray, dname: str = "X and Y"
) -> Dict[str, Any]:
    """
    Perform the Two-Sample Scott test for the equality of two covariance matrices from
    'Schott, J. R. (2007). A test for the equality of covariance matrices
    when the dimension is large relative to the sample size.'

    Parameters
    ----------
    X : np.ndarray
        The n1 x p data matrix for sample 1.
    Y : np.ndarray
        The n2 x p data matrix for sample 2.
    dname : str, optional
        The name of the data, by default "X and Y".

    Returns
    -------
    sc_res : dict
        A dictionary containing the test statistics, p-value,
        alternative hypothesis, method, and data names.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    p = X.shape[1]
    n1 = X.shape[0]
    n2 = Y.shape[0]

    if Y.shape[1] != p:
        raise ValueError("Different dimensions of X and Y.")

    Sx = np.cov(X, rowvar=False) * (n1 - 1) / n1
    Sy = np.cov(Y, rowvar=False) * (n2 - 1) / n2

    Sxx = Sx * n1 / (n1 - 1)
    Syy = Sy * n2 / (n2 - 1)

    SsS = (Sxx * n1 + Syy * n2) / (n1 + n2)

    eta1 = ((n1 - 1) + 2) * ((n1 - 1) - 1)
    eta2 = ((n2 - 1) + 2) * ((n2 - 1) - 1)
    d1 = (1 - (n1 - 1 - 2) / eta1) * np.sum(np.diag(Sxx @ Sxx))
    d2 = (1 - (n2 - 1 - 2) / eta2) * np.sum(np.diag(Syy @ Syy))
    d3 = 2 * np.sum(np.diag(Sxx @ Syy))
    d4 = (n1 - 1) / eta1 * np.sum(np.diag(Sxx)) ** 2
    d5 = (n2 - 1) / eta2 * np.sum(np.diag(Syy)) ** 2
    th = (
        4
        * (((n1 + n2 - 2) / ((n1 - 1) * (n2 - 1))) ** 2)
        * (
            (n1 + n2 - 2) ** 2
            / ((n1 + n2) * (n1 + n2 - 2 - 1))
            * (
                np.sum(np.diag(SsS @ SsS))
                - (np.sum(np.diag(SsS))) ** 2 / (n1 + n2 - 2)
            )
        )
        ** 2
    )
    Sc = (d1 + d2 - d3 - d4 - d5) / np.sqrt(th)

    sc_p = (1 - norm.cdf(np.abs(Sc))) * 2
    sc_res = {
        "statistics": np.abs(Sc),
        "p.value": sc_p,
        "alternative": "two.sided",
        "method": "Two-Sample Scott test",
        "data.name": dname,
    }

    return sc_res
