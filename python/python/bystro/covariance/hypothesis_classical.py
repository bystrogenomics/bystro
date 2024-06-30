import numpy as np
import numpy.linalg as la
import scipy.stats as stats  # type: ignore
from scipy.stats import chi2, norm  # type: ignore

#####################
#####################
##                 ##
## Spherical tests ##
##                 ##
#####################
#####################


def john_stat(data):
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    trace_S2 = np.trace(np.dot(sample_cov_matrix, sample_cov_matrix))
    U = (1 / p) * trace_S2 / ((1 / p) * trace_S) ** 2 - 1
    return U


def john_sphericity_test(data):
    n, p = data.shape
    U = john_stat(data)
    degree_of_freedom = p * (p + 1) / 2 - 1
    stat = U * n * p / 2
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


######################
######################
##                  ##
##  Identity tests  ##
##                  ##
######################
######################


def nagao_stat(data):
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    trace_S2 = np.trace(np.dot(sample_cov_matrix, sample_cov_matrix))
    V = 1 / p * trace_S2 - 2 / p * trace_S + 1
    return V


def nagao_identity_test(data):
    n, p = data.shape
    V = nagao_stat(data)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * V
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


def ledoit_wolf_stat(data):
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    SmI = sample_cov_matrix - np.eye(p)
    trace_smi2 = np.trace(np.dot(SmI, SmI))
    trace_S = np.trace(sample_cov_matrix)
    W = 1 / p * trace_smi2 - 1 / (n * p) * trace_S**2 + p / n
    return W


def ledoit_wolf_identity_test(data):
    n, p = data.shape
    W = ledoit_wolf_stat(data)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * W
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


#########################
#########################
##                     ##
##   One-sample tests  ##
##                     ##
#########################
#########################


def fisher_2012_stat_(n, p, S_):
    c = p / n
    ahat2 = (n**2 / ((n - 1) * (n + 2) * p)) * (
        np.sum(np.diag(S_ @ S_)) - (np.sum(np.diag(S_)) ** 2) / n
    )
    gamma = (n**5 * (n**2 + n + 2)) / (
        (n + 1) * (n + 2) * (n + 4) * (n + 6) * (n - 1) * (n - 2) * (n - 3)
    )
    ahat4 = (gamma / p) * (
        np.sum(np.diag(S_ @ S_ @ S_ @ S_))
        - (4 / n) * np.sum(np.diag(S_ @ S_ @ S_)) * np.sum(np.diag(S_))
        - ((2 * (n**2) + 3 * n - 6) / (n * (n**2 + n + 2)))
        * (np.sum(np.diag(S_ @ S_)) ** 2)
        + ((2 * (5 * n + 6)) / (n * (n**2 + n + 2)))
        * np.sum(np.diag(S_ @ S_))
        * (np.sum(np.diag(S_)) ** 2)
        - ((5 * n + 6) / ((n**2) * (n**2 + n + 2)))
        * (np.sum(np.diag(S_)) ** 4)
    )
    return (n / np.sqrt(8 * (c**2 + 12 * c + 8))) * (ahat4 - 2 * ahat2 + 1)


def fisher_single_sample_test(x, Sigma="identity"):
    p = x.shape[1]
    n = x.shape[0]
    S = np.cov(x, rowvar=False)

    if Sigma == "identity":
        S_ = S
    else:
        sv = la.svd(Sigma)
        svDf = la.svd(S)
        x_ = (
            svDf[0]
            @ np.diag(np.sqrt(sv[1]))
            @ la.inv(sv[0] @ np.diag(np.sqrt(sv[1])))
        )
        S_ = x_.T @ x_

    statistic = fisher_2012_stat_(n - 1, p, S_)
    p_value = 1 - norm.cdf(abs(statistic))

    return {
        "stat": statistic,
        "p_value": p_value,
    }


def srivastava2011_(n, p, S_):
    term1 = (
        (n**2 / ((n - 1) * (n + 2)))
        * (np.trace(S_ @ S_) - np.trace(S_) ** 2 / n)
        / p
    )
    term2 = 2 * (np.trace(S_) / p)
    return n * (term1 - term2 + 1) / 2


def srivastava2011_one_sample_test(x, Sigma="identity"):
    p = x.shape[1]
    n = x.shape[0]
    S = np.cov(x, rowvar=False)

    if Sigma == "identity":
        S_ = S
    else:
        sv = np.linalg.svd(Sigma)
        svDf = np.linalg.svd(S)
        x_ = (
            svDf[0]
            @ np.diag(np.sqrt(sv[1]))
            @ np.linalg.inv(sv[0] @ np.diag(np.sqrt(sv[1])))
        )
        S_ = x_.T @ x_

    statistic = srivastava2011_(n - 1, p, S_)
    p_value = 1 - norm.cdf(abs(statistic))

    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results


########################
########################
##                    ##
##  Two-sample tests  ##
##                    ##
########################
########################


def srivastava_yanagihara_stat(x):
    len_x = len(x)
    pmat = x[1]
    p = pmat.shape[1]
    ntot = 0
    ns = np.zeros(len_x)
    a2i = np.zeros(len_x)
    a1i = np.zeros(len_x)
    samplecov = []
    Apool = np.zeros((p, p))

    for i in range(len_x):
        mats = x[i]
        n = mats.shape[0]
        ns[i] = n
        covar = np.cov(mats, rowvar=False)
        samplecov.append(covar)

        covartrace = np.trace(covar)
        covar2trace = np.trace(covar @ covar)
        a2i[i] = ((n - 1) ** 2 / (p * (n - 2) * (n + 1))) * (
            covar2trace - (1.0 / (n - 1)) * covartrace**2
        )
        a1i[i] = covartrace / p

        ntot += n - 1
        Apool += covar * (n - 1)

    pooledCov = Apool / ntot
    pooledcov2trace = np.trace(pooledCov @ pooledCov)
    pooledcovtrace = np.trace(pooledCov)
    a2 = (ntot**2 / (p * (ntot - 1) * (ntot + 2))) * (
        pooledcov2trace - (1.0 / ntot) * pooledcovtrace**2
    )
    a1 = pooledcovtrace / p

    a3 = (1.0 / (ntot * (ntot**2 + 3 * ntot + 4))) * (
        np.trace(Apool @ Apool @ Apool) / p
        - 3.0 * ntot * (ntot + 1) * p * a2 * a1
        - ntot * p**2 * a1**3
    )

    c0 = ntot**4 + 6 * ntot**3 + 21 * ntot**2 + 18 * ntot
    c1 = 4 * ntot**3 + 12 * ntot**2 + 18 * ntot
    c2 = 6 * ntot**2 + 4 * ntot
    c3 = 2 * ntot**3 + 5 * ntot**2 + 7 * ntot

    a4 = (1.0 / c0) * (
        np.trace(Apool @ Apool @ Apool @ Apool) / p
        - c1 * a1
        - c2 * p * a1**2 * a2
        - c3 * a2**2
        - ntot * p**3 * a1**4
    )

    ksi2i = np.zeros(len_x)
    gammai = np.zeros(len_x)
    gammabarnum = 0
    gammabardem = 0

    for i in range(len_x):
        ksi2i[i] = (
            4.0
            / (ns[i] - 1) ** 2
            * (
                (a2**2 / a1**4)
                + 2.0
                * (ns[i] - 1)
                / p
                * ((a2**3 / a1**6) - 2.0 * a2 * a3 / a1**5 + a4 / a1**4)
            )
        )

        gammai[i] = a2i[i] / a1i[i] ** 2
        gammabarnum += gammai[i] / ksi2i[i]
        gammabardem += 1.0 / ksi2i[i]

    gammabar = gammabarnum / gammabardem

    stat = 0
    for i in range(len_x):
        stat += (gammai[i] - gammabar) ** 2 / ksi2i[i]

    return stat


def srivastavayanagihara_two_sample_test(x):
    matrix_ls = x

    # Compute the statistic
    statistic = srivastava_yanagihara_stat(matrix_ls)
    parameter = len(matrix_ls) - 1

    p_value = 1 - chi2.cdf(statistic, parameter)

    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results


def srivastava_2007_stat(x):
    len_x = len(x)
    pmat = x[1]
    p = pmat.shape[1]
    ntot = 0
    Apool = np.zeros((p, p))
    ns = np.zeros(len_x)
    a2i = np.zeros(len_x)
    samplecov = []

    for i in range(len_x):
        mats = x[i]
        n = mats.shape[0]
        ns[i] = n
        covar = np.cov(mats, rowvar=False)
        samplecov.append(covar)

        covar2trace = np.trace(covar @ covar)
        covartrace = np.trace(covar)
        a2i[i] = (
            (n - 1) ** 2
            / (p * (n - 2) * (n + 1))
            * (covar2trace - (1.0 / (n - 1)) * covartrace**2)
        )

        ntot += n - 1
        Apool += covar * (n - 1)

    pooledCov = Apool / ntot
    pooledcov2trace = np.trace(pooledCov @ pooledCov)
    pooledcovtrace = np.trace(pooledCov)
    a2 = (
        ntot**2
        / (p * (ntot - 1) * (ntot + 2))
        * (pooledcov2trace - (1.0 / ntot) * pooledcovtrace**2)
    )

    a1 = pooledcovtrace / p

    c0 = ntot**4 + 6 * ntot**3 + 21 * ntot**2 + 18 * ntot
    c1 = 4 * ntot**3 + 12 * ntot**2 + 18 * ntot
    c2 = 6 * ntot**2 + 4 * ntot
    c3 = 2 * ntot**3 + 5 * ntot**2 + 7 * ntot

    a4 = (1.0 / c0) * (
        np.trace(Apool @ Apool @ Apool @ Apool) / p
        - c1 * a1
        - c2 * a1**2 * a2
        - c3 * a2**2
        - ntot * a1**4 * p**3
    )

    eta2i = np.zeros(len_x)
    abarnum = 0
    abardem = 0

    for i in range(len_x):
        eta2i[i] = (
            4.0
            / (ns[i] - 1) ** 2
            * a2**2
            * (1.0 + 2.0 * (ns[i] - 1) * a4 / (p * a2**2))
        )

        abarnum += a2i[i] / eta2i[i]
        abardem += 1.0 / eta2i[i]

    abar = abarnum / abardem

    stat = 0
    for i in range(len_x):
        stat += (a2i[i] - abar) ** 2 / eta2i[i]

    return stat


def srivastava_two_sample_test(x):
    matrix_ls = x

    # Compute the statistic
    statistic = srivastava_2007_stat(matrix_ls)

    parameter = len(matrix_ls) - 1

    p_value = 1 - chi2.cdf(statistic, parameter)

    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results
