"""
This module provides functions for performing various statistical tests
related to covariance matrix analysis. It includes functions for computing
statistics and conducting tests such as the John sphericity test, Nagao
identity test, Ledoit-Wolf identity test, and multiple versions of tests
developed by Srivastava. Each function calculates a specific statistic and
performs a chi-squared test to determine p-values.

Functions:
- john_stat, john_sphericity_test: Evaluate sphericity of a covariance 
  matrix.
- nagao_stat, nagao_identity_test: Test for the covariance matrix being 
  the identity
- ledoit_wolf_stat, ledoit_wolf_identity_test: Implement Ledoit-Wolf's 
  method.
- fisher_2012_stat_, fisher_single_sample_test: Fisher's identity testing.
- srivastava2011_, srivastava2011_one_sample_test: Srivastava's 2011 tests.
- srivastava_yanagihara_stat, srivastavayanagihara_two_sample_test: 
  Two-sample tests adjusting for eigenvalue clusters.
- srivastava_2007_stat, srivastava_two_sample_test: Srivastava's two-sample
  covariance tests.
"""
import numpy as np
import scipy.stats as stats # type: ignore
from typing import Dict, Union, List


def john_stat(data: np.ndarray) -> float:
    """
    Calculate the John statistic for sphericity.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    float: Calculated John statistic.
    """
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    trace_S2 = np.trace(np.dot(sample_cov_matrix, sample_cov_matrix))
    U = (1 / p) * trace_S2 / ((1 / p) * trace_S) ** 2 - 1
    return U


def john_sphericity_test(data: np.ndarray) -> Dict[str, float]:
    """
    Perform the John sphericity test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    Dict[str, float]: Test results including the statistic and p-value.
    """
    n, p = data.shape
    U = john_stat(data)
    degree_of_freedom = p * (p + 1) / 2 - 1
    stat = U * n * p / 2
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


def nagao_stat(data: np.ndarray) -> float:
    """
    Calculate the Nagao statistic for identity test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    float: Calculated Nagao statistic.
    """
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    trace_S2 = np.trace(np.dot(sample_cov_matrix, sample_cov_matrix))
    V = 1 / p * trace_S2 - 2 / p * trace_S + 1
    return V


def nagao_identity_test(data: np.ndarray) -> Dict[str, float]:
    """
    Perform the Nagao identity test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    Dict[str, float]: Test results including the statistic and p-value.
    """
    n, p = data.shape
    V = nagao_stat(data)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * V
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


def ledoit_wolf_stat(data: np.ndarray) -> float:
    """
    Calculate the Ledoit-Wolf statistic for identity test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    float: Calculated Ledoit-Wolf statistic.
    """
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    SmI = sample_cov_matrix - np.eye(p)
    trace_smi2 = np.trace(np.dot(SmI, SmI))
    trace_S = np.trace(sample_cov_matrix)
    W = 1 / p * trace_smi2 - 1 / (n * p) * trace_S**2 + p / n
    return W


def ledoit_wolf_identity_test(data: np.ndarray) -> Dict[str, float]:
    """
    Perform the Ledoit-Wolf identity test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    Dict[str, float]: Test results including the statistic and p-value.
    """
    n, p = data.shape
    W = ledoit_wolf_stat(data)
    degree_of_freedom = p * (p + 1) / 2
    stat = n * p / 2 * W
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


def fisher_2012_stat_(data: np.ndarray) -> float:
    """
    Calculate the Fisher 2012 statistic for identity test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    float: Calculated Fisher 2012 statistic.
    """
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    trace_S = np.trace(sample_cov_matrix)
    trace_S2 = np.trace(np.dot(sample_cov_matrix, sample_cov_matrix))
    T = (n - 1) * (trace_S2 / (trace_S**2) - 1)
    return T


def fisher_single_sample_test(data: np.ndarray) -> Dict[str, float]:
    """
    Perform the Fisher single sample test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    Dict[str, float]: Test results including the statistic and p-value.
    """
    n, p = data.shape
    T = fisher_2012_stat_(data)
    degree_of_freedom = p * (p + 1) / 2 - 1
    stat = T
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


def srivastava2011_(data: np.ndarray) -> float:
    """
    Calculate the Srivastava 2011 statistic for identity test.

    Parameters:
    data (np.ndarray): Input data matrix.

    Returns:
    float: Calculated Srivastava 2011 statistic.
    """
    n, p = data.shape
    sample_cov_matrix = np.cov(data, rowvar=False)
    mean_diag = np.mean(np.diag(sample_cov_matrix))
    T = (
        (n - 1)
        * (p * (p + 1) / 2)
        * ((mean_diag / np.trace(sample_cov_matrix)) ** 2 - 1)
    )
    return T


def srivastava2011_one_sample_test(
    x: np.ndarray, Sigma: Union[str, np.ndarray] = "identity"
) -> Dict[str, float]:
    """
    Perform the Srivastava 2011 one sample test.

    Parameters:
    x (np.ndarray): Input data matrix.
    Sigma (Union[str, np.ndarray]): Covariance matrix or 'identity' for identity matrix.

    Returns:
    Dict[str, float]: Test results including the statistic and p-value.
    """
    n, p = x.shape
    if Sigma == "identity":
        Sigma = np.eye(p)
    elif isinstance(Sigma, str):
        raise ValueError("Sigma must be 'identity' or a numpy array.")
    T = srivastava2011_(x)
    degree_of_freedom = p * (p + 1) / 2 - 1
    stat = T
    p_value = 1 - stats.chi2.cdf(stat, degree_of_freedom)
    results = {"stat": stat, "p_value": p_value}
    return results


def srivastava_yanagihara_stat(x: List[np.ndarray]) -> float:
    """
    Calculate the Srivastava-Yanagihara statistic for two sample test.

    Parameters:
    x (List[np.ndarray]): List of input data matrices.

    Returns:
    float: Calculated Srivastava-Yanagihara statistic.
    """
    len_x = len(x)
    p = x[0].shape[1]
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


def srivastavayanagihara_two_sample_test(
    x: List[np.ndarray],
) -> Dict[str, float]:
    """
    Perform the Srivastava-Yanagihara two sample test.

    Parameters:
    x (List[np.ndarray]): List of input data matrices.

    Returns:
    Dict[str, float]: Test results including the statistic and p-value.
    """
    statistic = srivastava_yanagihara_stat(x)
    parameter = len(x) - 1
    p_value = 1 - stats.chi2.cdf(statistic, parameter)
    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results


def srivastava_2007_stat(x: List[np.ndarray]) -> float:
    """
    Calculate the Srivastava 2007 statistic for two sample test.

    Parameters:
    x (List[np.ndarray]): List of input data matrices.

    Returns:
    float: Calculated Srivastava 2007 statistic.
    """
    len_x = len(x)
    p = x[0].shape[1]
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


def srivastava_two_sample_test(x: List[np.ndarray]) -> Dict[str, float]:
    """
    Perform the Srivastava two sample test.

    Parameters:
    x (List[np.ndarray]): List of input data matrices.

    Returns:
    Dict[str, float]: Test results including the statistic and p-value.
    """
    matrix_ls = x
    statistic = srivastava_2007_stat(matrix_ls)
    parameter = len(matrix_ls) - 1
    p_value = 1 - stats.chi2.cdf(statistic, parameter)
    results = {
        "stat": statistic,
        "p_value": p_value,
    }
    return results
