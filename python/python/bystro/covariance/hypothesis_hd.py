import numpy as np
from scipy.linalg import eigh  # type: ignore
from scipy.stats import norm, chi2  # type: ignore
from scipy.sparse.linalg import eigs  # type: ignore
from statsmodels.stats.multitest import multipletests

rng = np.random.default_rng()


def two_way_sampling(X: np.ndarray, Y: np.ndarray, n: int) -> dict:
    """
    Perform two-way sampling on matrices X and Y.

    Parameters:
    X (np.ndarray): First matrix.
    Y (np.ndarray): Second matrix.
    n (int): Sample size.

    Returns:
    dict: Sampled matrices Xs, Ys, Zs.
    """
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

    Parameters:
    X (np.ndarray): Input matrix.

    Returns:
    np.ndarray: Eigenvalues of the sample covariance matrix.
    """
    n, p = X.shape
    X1 = X - np.mean(X, axis=0)
    return eigh(np.dot(X1.T, X1) / np.sqrt(n * p), eigvals_only=True)


def k(x: float) -> float:
    """
    Kernel function K.

    Parameters:
    x (float): Input value.

    Returns:
    float: Output of the kernel function.
    """
    if abs(x) >= 1.05:
        return 0
    if abs(x) <= 1:
        return 1
    return np.exp(1 / 0.05**2 - 1 / (0.05**2 - (abs(x) - 1) ** 2))


def t(lambda_: np.ndarray, gamma: float, eta0: float) -> float:
    """
    Compute T value.

    Parameters:
    lambda_ (np.ndarray): Array of eigenvalues.
    gamma (float): Gamma value.
    eta0 (float): Eta value.

    Returns:
    float: T value.
    """
    return sum(
        (lambda_i - gamma) / eta0 * k((lambda_i - gamma) / eta0)
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

    Parameters:
    gamma (float): Gamma value.
    lambda1 (np.ndarray): First array of eigenvalues.
    lambda2 (np.ndarray): Second array of eigenvalues.
    epsilon (float): Tolerance value.

    Returns:
    bool: True if efficient, False otherwise.
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
    thres: float = None,
    mode: str = "test",
) -> dict:
    """
    Perform a two-sample test.

    Parameters:
    X (np.ndarray): First matrix.
    Y (np.ndarray): Second matrix.
    n (int): Sample size.
    const (float): Constant value.
    alpha (float): Significance level.
    epsilon (float): Tolerance value.
    thres (float): Threshold value.
    mode (str): Mode of the test.

    Returns:
    dict: Result of the two-sample test.
    """
    sample_list = two_way_sampling(X, Y, n)
    Xs, Ys, Zs = sample_list["Xs"], sample_list["Ys"], sample_list["Zs"]

    eig_xs = cov_eigs(Xs)
    eig_ys = cov_eigs(Ys)
    eig_zs = cov_eigs(Zs)
    gamma = np.median(eig_zs)
    if not check_efficient(gamma, eig_xs, eig_ys, epsilon):
        return {"efficient": False, "c": 1}

    gamma = np.median(eig_zs)
    eta0 = np.std(eig_zs) * const

    Tx = t(eig_xs, gamma, eta0)
    Ty = t(eig_ys, gamma, eta0)

    if mode != "test":
        return abs(Tx - Ty)

    if thres is None:
        thres = 2.6

    threshold = thres / norm.ppf(1 - 0.05 / 2) * norm.ppf(1 - alpha / 2)
    result = (Tx - Ty) / (thres / norm.ppf(1 - 0.05 / 2))
    return {
        "efficient": True,
        "c": int(abs(Tx - Ty) > threshold),
        "statistic": result,
    }


def two_sample_test(
    X: np.ndarray,
    Y: np.ndarray,
    n: int = None,
    k: int = 100,
    const: float = None,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    thres: float = None,
    calib: bool = False,
) -> dict:
    """
    Perform a two-sample test multiple times.

    Parameters:
    X (np.ndarray): First matrix.
    Y (np.ndarray): Second matrix.
    n (int): Sample size.
    k (int): Number of iterations.
    const (float): Constant value.
    alpha (float): Significance level.
    epsilon (float): Tolerance value.
    thres (float): Threshold value.
    calib (bool): Calibration flag.

    Returns:
    dict: Result of the two-sample test.
    """
    reject, df, statistic = 0, 0, 0

    if n is None:
        n1, n2 = X.shape[0], Y.shape[0]
        n = min(max(n1, n2) / 2, n1, n2) - 5

    if thres is None:
        thres = (
            2.6
            if not calib
            else calibration(
                n1=X.shape[0],
                n2=Y.shape[0],
                p=X.shape[1],
                n=n,
                alpha=alpha,
                const=0.5,
                iterations=100,
            )
        )

    if const is None:
        const = c_tuning(X, Y, n, alpha=alpha, epsilon=epsilon, thres=thres)[
            "c"
        ]

    for _ in range(k):
        result = two_sample_test_(
            X, Y, n, alpha=alpha, const=const, epsilon=epsilon, thres=thres
        )
        if result["efficient"]:
            df += 1
            statistic += result["statistic"] ** 2
        if result["c"] == 1:
            reject += 1

    threshold = rng.binomial(k, alpha)

    return {"decision": int(reject > threshold), "df": df, "reject": reject}


def calibration(
    n1: int,
    n2: int,
    p: int,
    n: int,
    alpha: float = 0.05,
    const: float = 0.5,
    iterations: int = 100,
    K: int = 100,
) -> float:
    """
    Perform calibration to find threshold.

    Parameters:
    n1 (int): Size of the first sample.
    n2 (int): Size of the second sample.
    p (int): Number of columns.
    n (int): Sample size.
    alpha (float): Significance level.
    const (float): Constant value.
    iterations (int): Number of iterations.
    K (int): Number of sub-iterations.

    Returns:
    float: Threshold value.
    """
    values = []
    for _ in range(iterations):
        X = rng.normal(0, 1, (n1, p))
        Y = rng.normal(0, 1, (n2, p))
        for _ in range(K):
            values.append(
                two_sample_test_(
                    X, Y, n, alpha=alpha, const=const, mode="calib"
                )
            )
    return np.quantile(values, 1 - alpha)


def c_tuning(
    X: np.ndarray,
    Y: np.ndarray,
    n: int,
    thres: float = None,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    K: int = 500,
) -> dict:
    """
    Tune the constant C.

    Parameters:
    X (np.ndarray): First matrix.
    Y (np.ndarray): Second matrix.
    n (int): Sample size.
    thres (float): Threshold value.
    alpha (float): Significance level.
    epsilon (float): Tolerance value.
    K (int): Number of iterations.

    Returns:
    dict: Tuned constant C and rates.
    """
    Cs = np.arange(1, 31) / 10
    rates = []

    if thres is None:
        thres = 2.6

    for c in Cs:
        all_ = 0
        rej = 0
        for _ in range(K):
            result = two_sample_test_(
                X, Y, n, alpha=alpha, const=c, epsilon=epsilon, thres=thres
            )
            if result["efficient"]:
                all_ += 1
                rej += result["c"]
        rates.append(rej / all_ if all_ > 50 else 1)

    stable = find_stable(rates)
    return {"c": Cs[stable], "rates": rates}


def find_stable(xs: list) -> int:
    """
    Find the stable point in a list of rates.

    Parameters:
    xs (list): List of rates.

    Returns:
    int: Index of the stable point.
    """
    roll_average = np.convolve(xs, np.ones(3) / 3, mode="valid")
    vars_ = [np.var(roll_average[:i]) for i in range(2, len(roll_average))]
    for i in range(len(vars_) - 1):
        if vars_[i + 1] < vars_[i] and roll_average[i] > max(roll_average) / 5:
            break
    return i + 2


def movevar(xs: list) -> list:
    """
    Compute the moving variance of a list.

    Parameters:
    xs (list): List of values.

    Returns:
    list: List of variances.
    """
    n = len(xs)
    vars_ = []
    for i in range(2, n + 1):
        vars_.append(np.var(xs[:i]))
    return vars_


def clx2013(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Perform the CLX2013 test.

    Parameters:
    X (np.ndarray): First matrix.
    Y (np.ndarray): Second matrix.

    Returns:
    dict: Test statistic and p-value.
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

    Parameters:
    X (np.ndarray): First matrix.
    Y (np.ndarray): Second matrix.

    Returns:
    dict: Test statistic and p-value.
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
    X: np.ndarray, Y: np.ndarray, N: int = None, alpha: float = 0.05
) -> dict:
    """
    Perform the HC2018 test.

    Parameters:
    X (np.ndarray): First matrix.
    Y (np.ndarray): Second matrix.
    N (int): Parameter for the test.
    alpha (float): Significance level.

    Returns:
    dict: Test result.
    """
    if N is None:
        N = int(np.floor(X.shape[1] ** 0.7))

    def double_sum(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.sum(X1, axis=0) * np.sum(X2, axis=0) - np.sum(X1 * X2, axis=0)

    def triple_sum(
        X1: np.ndarray, X2: np.ndarray, X3: np.ndarray
    ) -> np.ndarray:
        return (
            double_sum(X1, X2) * np.sum(X3, axis=0)
            - double_sum(X1 * X3, X2)
            - double_sum(X1, X2 * X3)
        )

    def quad_sum(
        X1: np.ndarray, X2: np.ndarray, X3: np.ndarray, X4: np.ndarray
    ) -> np.ndarray:
        return (
            triple_sum(X1, X2, X3) * np.sum(X4, axis=0)
            - triple_sum(X1 * X4, X2, X3)
            - triple_sum(X1, X2 * X4, X3)
            - triple_sum(X1, X2, X3 * X4)
        )

    def di(X: np.ndarray, q: int) -> float:
        n, p = X.shape
        X1 = X[:, : p - q]
        X2 = X[:, q:]
        D_1 = np.sum(double_sum(X1 * X2, X1 * X2))
        D_2 = np.sum(triple_sum(X1, X2, X1 * X2))
        D_3 = np.sum(quad_sum(X1, X2, X1, X2))
        return (
            1 / (n * (n - 1)) * D_1
            - 2 / (n * (n - 1) * (n - 1)) * D_2
            + 1 / (n * (n - 1) * (n - 2) * (n - 3)) * D_3
        )

    def dc(X1: np.ndarray, X2: np.ndarray, q: int) -> float:
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
        return (
            Dc_1 / (n1 * n2)
            - Dc_2 / (n1 * (n1 - 1) * n2)
            - Dc_3 / (n1 * n2 * (n2 - 1))
            + Dc_4 / (n1 * (n1 - 1) * n2 * (n2 - 1))
        )

    def sq(X1: np.ndarray, X2: np.ndarray, q: int) -> float:
        return di(X1, q) + di(X2, q) - 2 * dc(X1, X2, q)

    def ri(X: np.ndarray, q: int) -> float:
        n, p = X.shape

        X = X - np.mean(X, axis=0)

        X1 = X[:, : p - q]
        X2 = X[:, q:]

        Y = X1 * X2

        Y -= np.mean(Y, axis=0) / (n - 1)

        YYt2 = np.dot(Y, Y.T) ** 2

        result = (np.sum(YYt2) - np.sum(np.diag(YYt2))) / (n * (n - 1))

        return result

    def rc(X1: np.ndarray, X2: np.ndarray, q: int) -> float:
        n1, p = X1.shape
        X1 = X1 - np.mean(X1, axis=0)

        X11 = X1[:, : p - q]

        X12 = X1[:, q:]

        Y1 = X11 * X12

        Y1 -= np.mean(Y1, axis=0) / (n1 - 1)

        n2 = X2.shape[0]
        X2 = X2 - np.mean(X2, axis=0)

        X21 = X2[:, : p - q]

        X22 = X2[:, q:]

        Y2 = X21 * X22

        Y2 -= np.mean(Y2, axis=0) / (n2 - 1)

        result = np.sum((np.dot(Y1, Y2.T)) ** 2) / (n1 * n2)

        return result

    def v2(X1: np.ndarray, X2: np.ndarray, q: int) -> float:
        n1, n2 = X1.shape[0], X2.shape[0]
        return (
            ri(X1, q) * 2 / (n1 * (n1 - 1))
            + ri(X2, q) * 2 / (n2 * (n2 - 1))
            + rc(X1, X2, q) * 4 / (n1 * n2)
        )

    def one_super(X1: np.ndarray, X2: np.ndarray, q: int) -> float:
        chi = sq(X1, X2, q) ** 2 / v2(X1, X2, q)
        return chi2.sf(chi, 1)

    pvalues = [one_super(X, Y, i) for i in range(N + 1)]
    test = adaptive_sts(pvalues, alpha=alpha, lambda_=0.5, silent=True)

    return {"reject": np.sum(test["rejected"]), "pvalues": pvalues, "N": N}


def storey_pi0_est(pValues: list, lambda_: float = 0.5) -> dict:
    """
    Estimate the proportion of true null hypotheses.

    Parameters:
    pValues (list): List of p-values.
    lambda_ (float): Lambda parameter.

    Returns:
    dict: Estimated pi0.
    """
    pi0 = np.mean(np.array(pValues) > lambda_) / (1 - lambda_)
    pi0 = min(pi0, 1.0)
    return {"pi0": pi0}


def print_rejected(rejected: list, pValues: list, adjPValues: list) -> None:
    """
    Print the rejected hypotheses.

    Parameters:
    rejected (list): List of rejected hypotheses.
    pValues (list): List of p-values.
    adjPValues (list): List of adjusted p-values.
    """
    for i in range(len(pValues)):
        if rejected[i]:
            print(
                f"pValue: {pValues[i]}, adjPValue: {adjPValues[i]}, Rejected: {rejected[i]}"
            )


def adaptive_sts(
    pValues: list, alpha: float, lambda_: float = 0.5, silent: bool = False
) -> dict:
    """
    Perform the adaptive step-up procedure.

    Parameters:
    pValues (list): List of p-values.
    alpha (float): Significance level.
    lambda_ (float): Lambda parameter.
    silent (bool): Silent flag.

    Returns:
    dict: Result of the adaptive step-up procedure.
    """
    m = len(pValues)
    _, adjP, _, _ = multipletests(pValues, alpha=alpha, method="fdr_bh")

    pi0 = storey_pi0_est(pValues, lambda_)["pi0"]
    criticalValues = [(i * alpha) / (m * pi0) for i in range(1, m + 1)]
    adjPValues = adjP * min(pi0, 1)
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
