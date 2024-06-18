import numpy as np
import numpy.linalg as la
import pandas as pd
import math

from bystro.covariance.covariance_cov_shrinkage import (
    LinearInverseShrinkage,
    QuadraticInverseShrinkage,
    GeometricInverseShrinkage,
    qis,
    gis,
    lis,
)

pd.options.future.infer_string = True # type: ignore

def gis_original(Y, k=None): # type: ignore
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    c = p / n
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample = (sample + sample.T) / 2
    lambda1, u = np.linalg.eig(sample)
    lambda1 = lambda1.real
    u = u.real
    lambda1 = lambda1.real.clip(min=0)

    dfu = pd.DataFrame(u, columns=lambda1)
    dfu = dfu.sort_index(axis=1)
    lambda1 = dfu.columns # type: ignore
    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1 : p]

    dfl = pd.DataFrame()
    dfl["lambda"] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]
    Lj = pd.DataFrame(Lj.to_numpy())
    Lj_i = Lj.subtract(Lj.T)
    theta = (
        Lj.multiply(Lj_i)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )
    Htheta = (
        Lj.multiply(Lj * h)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )
    Atheta2 = theta**2 + Htheta**2

    if p <= n:
        deltahat_1 = (1 - c) * invlambda + 2 * c * invlambda * theta
        delta = 1 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )
        delta = delta.to_numpy()
    else:
        print("p must be <= n for the Symmetrized Kullback-Leibler divergence")
        return -1
    temp = pd.DataFrame(deltahat_1)
    x = min(invlambda)
    temp.loc[temp[0] < x, 0] = x
    deltaLIS_1 = temp[0]
    temp1 = dfu.to_numpy()
    temp2 = np.diag((delta / deltaLIS_1) ** 0.5)
    temp3 = dfu.T.to_numpy().conjugate()
    sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
    return sigmahat


def lis_original(Y, k=None): # type: ignore
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    c = p / n
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample = (sample + sample.T) / 2
    lambda1, u = np.linalg.eig(sample)
    lambda1 = lambda1.real
    u = u.real
    lambda1 = lambda1.real.clip(min=0)
    dfu = pd.DataFrame(u, columns=lambda1)
    dfu = dfu.sort_index(axis=1)
    lambda1 = dfu.columns # type: ignore
    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1 : p]

    dfl = pd.DataFrame()
    dfl["lambda"] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]
    Lj = pd.DataFrame(Lj.to_numpy())
    Lj_i = Lj.subtract(Lj.T)

    theta = (
        Lj.multiply(Lj_i)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )
    if p <= n:
        deltahat_1 = (1 - c) * invlambda + 2 * c * invlambda * theta
    else:
        print("p must be <= n for Stein's loss")
        return -1
    temp = pd.DataFrame(deltahat_1)
    x = min(invlambda)
    temp.loc[temp[0] < x, 0] = x
    deltaLIS_1 = temp[0]
    temp1 = dfu.to_numpy()
    temp2 = np.diag(1 / deltaLIS_1)
    temp3 = dfu.T.to_numpy().conjugate()
    sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))
    return sigmahat


def qis_original(Y, k=None): # type: ignore
    # Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    # Post-Condition: Sigmahat dataframe is returned

    # Set df dimensions
    N = Y.shape[0]  # num of columns
    p = Y.shape[1]  # num of rows

    # default setting
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)  # demean
        k = 1

    # vars
    n = N - k  # adjust effective sample size
    c = p / n  # concentration ratio

    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample = (sample + sample.T) / 2  # make symmetrical

    # Spectral decomp
    lambda1, u = np.linalg.eig(sample)  # use LAPACK routines
    lambda1 = lambda1.real  # clip imaginary part due to rounding error
    u = u.real  # clip imaginary part for eigenvectors

    lambda1 = lambda1.real.clip(min=0)  # reset negative values to 0
    dfu = pd.DataFrame(u, columns=lambda1)  # create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1)
    lambda1 = dfu.columns  # type: ignore 

    # COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35  # smoothing parameter
    invlambda = (
        1 / lambda1[max(1, p - n + 1) - 1 : p]
    )  # inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl["lambda"] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]  # like  1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())  # Reset column names
    Lj_i = Lj.subtract(Lj.T)  # like (1/lambda_j)-(1/lambda_i)

    theta = (
        Lj.multiply(Lj_i)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )  # smoothed Stein shrinker
    Htheta = (
        Lj.multiply(Lj * h)
        .div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h**2))
        .mean(axis=0)
    )  # its conjugate
    Atheta2 = theta**2 + Htheta**2  # its squared amplitude

    if p <= n:  # case where sample covariance matrix is not singular
        delta = 1 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )  # optimally shrunk eigenvalues
        delta = delta.to_numpy()
    else:
        delta0 = 1 / (
            (c - 1) * np.mean(invlambda.to_numpy()) # type: ignore
        )  # shrinkage of null
        #                                                 eigenvalues
        delta = np.repeat(delta0, p - n)
        delta = np.concatenate((delta, 1 / (invlambda * Atheta2)), axis=None)

    deltaQIS = delta * (sum(lambda1) / sum(delta))  # preserve trace

    temp1 = dfu.to_numpy()
    temp2 = np.diag(deltaQIS)
    temp3 = dfu.T.to_numpy().conjugate()
    sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1, temp2), temp3))

    return sigmahat


def test_linear_inverse_shrinkage(): # type: ignore
    rng = np.random.default_rng(2021)
    L = 80
    X = rng.normal(size=(20000, L))
    lis_instance = LinearInverseShrinkage()
    lis_instance.fit(X)
    val = la.norm(lis_instance.covariance - np.eye(L)) / la.norm(np.eye(L)) # type: ignore
    assert val < 0.01


def test_quadratic_inverse_shrinkage(): # type: ignore
    rng = np.random.default_rng(2021)
    L = 200
    X = rng.normal(size=(200000, L))
    qis_instance = QuadraticInverseShrinkage()
    qis_instance.fit(X)
    assert ( # type: ignore
        la.norm(qis_instance.covariance - np.eye(L)) / la.norm(np.eye(L)) < 0.01 # type: ignore
    ) # type: ignore


def test_geometric_inverse_shrinkage(): # type: ignore
    rng = np.random.default_rng(2021)
    L = 200
    X = rng.normal(size=(200000, L))
    gis_instance = GeometricInverseShrinkage()
    gis_instance.fit(X)
    assert ( # type: ignore
        la.norm(gis_instance.covariance - np.eye(L)) / la.norm(np.eye(L)) < 0.01 # type: ignore
    ) # type: ignore


def test_lis_function(): # type: ignore
    rng = np.random.default_rng(2021)
    L = 200
    Y = rng.normal(size=(10000, L))
    result = lis(Y)
    result2 = lis_original(pd.DataFrame(Y))
    assert result.shape == (L, L), "Output is covariance"
    assert la.norm(result - result2.to_numpy()) < 0.01, "Same result"


def test_qis_function(): # type: ignore
    rng = np.random.default_rng(2021)
    L = 40
    Y = rng.normal(size=(15000, L))
    result = qis(Y)
    result2 = qis_original(pd.DataFrame(Y))

    assert result.shape == (L, L), "Output is covariance"
    assert la.norm(result - result2.to_numpy()) < 0.01, "Same result"


def test_gis_function(): # type: ignore
    rng = np.random.default_rng(2021)
    L = 50
    Y = rng.normal(size=(15000, L))
    result = gis(Y)
    result2 = gis_original(pd.DataFrame(Y))
    r2 = result2.to_numpy()
    assert result.shape == (L, L), "Output is covariance"
    assert la.norm(r2 - result) < 0.01, "Identical outputs"
