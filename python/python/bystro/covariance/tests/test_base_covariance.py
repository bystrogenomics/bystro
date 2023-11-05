import numpy as np
import numpy.linalg as la
import scipy.stats as st
from bystro.covariance._base_covariance import (
    BaseCovariance,
    _score_samples,
    _conditional_score_sherman_woodbury,
    _conditional_score_samples_sherman_woodbury,
    _marginal_score_sherman_woodbury,
    _marginal_score_samples_sherman_woodbury,
    _score_sherman_woodbury,
    _score_samples_sherman_woodbury,
    inv_sherman_woodbury_fa,
    inv_sherman_woodbury_full,
    ldet_sherman_woodbury_fa,
    ldet_sherman_woodbury_full,
    _get_conditional_parameters_sherman_woodbury,
    _get_conditional_parameters,
)


def test_get_stable_rank():
    model = BaseCovariance()

    model.covariance = np.eye(10)
    assert model.get_stable_rank() == 10

    model.covariance = np.zeros((10, 10))
    model.covariance[0, 0] = 1
    assert model.get_stable_rank() == 1


def test_predict():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    cov = 1 / X.shape[0] * np.dot(X.T, X)
    model.covariance = cov
    idxs = np.ones(10)
    idxs[8:] = 0
    preds = model.predict(X, idxs)
    csub = cov[idxs == 1]
    c22 = csub[:, idxs == 1]
    c21 = csub[:, idxs == 0]
    csub = cov[idxs == 0]
    beta_bar = la.solve(c22, c21)
    mu = np.dot(X[:, idxs == 1], beta_bar)
    assert np.all(np.abs(mu - preds) < 1e-8)


def test_conditional_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    cov = 1 / X.shape[0] * np.dot(X.T, X)
    model.covariance = cov
    idxs = np.ones(10)
    idxs[8:] = 0
    score_samples = model.conditional_score_samples(X, idxs)
    csub = cov[idxs == 1]
    c22 = csub[:, idxs == 1]
    c21 = csub[:, idxs == 0]
    csub = cov[idxs == 0]
    c11 = csub[:, idxs == 0]
    beta_bar = la.solve(c22, c21)
    second = np.dot(c21.T, beta_bar)
    cov_bar = c11 - second
    mu = np.dot(X[:, idxs == 1], beta_bar)
    score_samples2 = _score_samples(cov_bar, X[:, idxs == 0] - mu)
    assert np.all(np.abs(score_samples - score_samples2) < 1e-8)


def test_conditional_score_samples_sherman_woodbury():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    W = rng.normal(size=(3, 10))
    Lambda = np.diag(np.abs(rng.normal(size=10)))
    cov = Lambda + np.dot(W.T, W)
    model = BaseCovariance()
    model.covariance = cov
    idxs = np.ones(10)
    idxs[8:] = 0
    score_samples = model.conditional_score_samples(X, idxs)
    score_samples_sw = _conditional_score_samples_sherman_woodbury(
        Lambda, W, X, idxs
    )
    assert np.all(np.abs(score_samples - score_samples_sw) < 1e-5)


def test_conditional_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    cov = 1 / X.shape[0] * np.dot(X.T, X)
    model.covariance = cov
    idxs = np.ones(10)
    idxs[8:] = 0
    score = model.conditional_score(X, idxs)
    csub = cov[idxs == 1]
    c22 = csub[:, idxs == 1]
    c21 = csub[:, idxs == 0]
    csub = cov[idxs == 0]
    c11 = csub[:, idxs == 0]
    beta_bar = la.solve(c22, c21)
    second = np.dot(c21.T, beta_bar)
    cov_bar = c11 - second
    mu = np.dot(X[:, idxs == 1], beta_bar)
    score_samples2 = _score_samples(cov_bar, X[:, idxs == 0] - mu)
    assert np.abs(np.mean(score_samples2) - score) < 1e-8


def test_conditional_score_sherman_woodbury():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    W = rng.normal(size=(3, 10))
    Lambda = np.diag(np.abs(rng.normal(size=10)))
    cov = Lambda + np.dot(W.T, W)
    model = BaseCovariance()
    model.covariance = cov
    idxs = np.ones(10)
    idxs[8:] = 0
    score = model.conditional_score(X, idxs)
    score_sw = _conditional_score_sherman_woodbury(Lambda, W, X, idxs)
    assert np.abs(score_sw - score) < 1e-8


def test_marginal_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(150, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    score_samples = model.marginal_score_samples(X[:, idxs == 1], idxs)
    mvn = st.multivariate_normal(mean=np.zeros(8), cov=model.covariance[:8, :8])
    logpdf = mvn.logpdf(X[:, :8])
    assert np.all(np.abs(logpdf - score_samples) < 1e-8)


def test_marginal_score_samples_sherman_woodbury():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(150, 10))
    W = rng.normal(size=(3, 10))
    Lambda = np.diag(np.abs(rng.normal(size=10)))
    model = BaseCovariance()
    model.covariance = Lambda + np.dot(W.T, W)
    idxs = np.ones(10)
    idxs[8:] = 0
    score_samples = model.marginal_score_samples(X[:, idxs == 1], idxs)
    score_samples_sw = _marginal_score_samples_sherman_woodbury(
        Lambda, W, X[:, idxs == 1], idxs
    )
    assert np.all(np.abs(score_samples_sw - score_samples) < 1e-8)


def test_marginal_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(15, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    score = model.marginal_score(X[:, idxs == 1], idxs)
    mvn = st.multivariate_normal(mean=np.zeros(8), cov=model.covariance[:8, :8])
    logpdf = mvn.logpdf(X[:, :8])
    assert np.mean(logpdf) == score


def test_marginal_score_sherman_woodbury():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(150, 10))
    W = rng.normal(size=(3, 10))
    Lambda = np.diag(np.abs(rng.normal(size=10)))
    model = BaseCovariance()
    model.covariance = Lambda + np.dot(W.T, W)
    idxs = np.ones(10)
    idxs[8:] = 0
    score = model.marginal_score(X[:, idxs == 1], idxs)
    score_sw = _marginal_score_sherman_woodbury(
        Lambda, W, X[:, idxs == 1], idxs
    )
    assert np.abs(score - score_sw) < 1e-6


def test_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    score = model.score(X, weights=1.0 * np.ones(1000))
    mvn = st.multivariate_normal(mean=np.zeros(10), cov=model.covariance)
    logpdf = mvn.logpdf(X)
    assert np.mean(logpdf) == score


def test_score_sherman_woodbury():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    W = rng.normal(size=(3, 10))
    Lambda = np.diag(np.abs(rng.normal(size=10)))
    model = BaseCovariance()
    model.covariance = Lambda + np.dot(W.T, W)
    weights = 1.0 * np.ones(1000)
    score = model.score(X, weights=weights)
    score_sw = _score_sherman_woodbury(Lambda, W, X, weights=weights)
    assert np.abs(score - score_sw) < 1e-6


def test_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    score_samples = model.score_samples(X)
    mvn = st.multivariate_normal(mean=np.zeros(10), cov=model.covariance)
    logpdf = mvn.logpdf(X)
    assert np.all(np.abs(logpdf - score_samples) < 1e-8)


def test_score_samples_sherman_woodbury():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    W = rng.normal(size=(3, 10))
    Lambda = np.diag(np.abs(rng.normal(size=10)))
    model = BaseCovariance()
    model.covariance = Lambda + np.dot(W.T, W)
    score_samples = model.score_samples(X)
    score_samples_sw = _score_samples_sherman_woodbury(Lambda, W, X)
    assert np.all(np.abs(score_samples_sw - score_samples) < 1e-8)


def test_entropy():
    model = BaseCovariance()
    model.covariance = np.eye(10)
    _, logdet = la.slogdet(model.covariance)
    ent = 10 / 2 * np.log(2 * np.pi * np.exp(1)) + 1 / 2 * logdet
    entropy = model.entropy()
    assert np.abs(ent - entropy) < 1e-8


def test_entropy_subset():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    ent_sub = model.entropy_subset(idxs)
    _, logdet = la.slogdet(model.covariance[:8, :8])
    ent = 8 / 2 * np.log(2 * np.pi * np.exp(1)) + 1 / 2 * logdet
    assert np.abs(ent_sub - ent) < 1e-8


def test_mutual_information():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs1 = np.ones(10)
    idxs2 = np.ones(10)
    idxs1[5:] = 0
    idxs2[:5] = 0
    mi = model.mutual_information(idxs1, idxs2)
    _, ldet1 = la.slogdet(model.covariance[:5, :5])
    _, ldet2 = la.slogdet(model.covariance[5:, 5:])
    _, ldet = la.slogdet(model.covariance)
    mi_est = 0.5 * (ldet1 + ldet2 - ldet)
    assert np.abs(mi_est - mi) < 1e-5


def test_inv_sherman_woodbury_fa():
    rng = np.random.default_rng(2021)
    p = 20
    Lambda = np.diag(rng.gamma(1, 1, size=p))
    W = rng.normal(size=(3, p))
    Sigma = Lambda + np.dot(W.T, W)
    Sigma_inv = la.inv(Sigma)
    Sigma_inv_woodbury = inv_sherman_woodbury_fa(Lambda, W)
    assert np.sum(np.abs(Sigma_inv - Sigma_inv_woodbury)) < 1e-5


def test_inv_sherman_woodbury_full():
    rng = np.random.default_rng(2021)
    p = 20
    L = 4
    A = rng.normal(size=(p, p))
    U = rng.normal(size=(p, L))
    B = rng.normal(size=(L, L))
    V = rng.normal(size=(L, p))
    UBV = np.dot(np.dot(U, B), V)
    AUBV = A + UBV
    AUBVi = la.inv(AUBV)
    AUBVi_woodbury = inv_sherman_woodbury_full(A, U, B, V)
    assert np.sum(np.abs(AUBVi - AUBVi_woodbury)) < 1e-5


def test_ldet_sherman_woodbury_fa():
    rng = np.random.default_rng(2021)
    p = 20
    Lambda = np.diag(rng.gamma(1, 1, size=p))
    W = rng.normal(size=(3, p))
    Sigma = Lambda + np.dot(W.T, W)
    _, ldet = la.slogdet(Sigma)
    ldet_woodbury = ldet_sherman_woodbury_fa(Lambda, W)
    assert np.abs(ldet - ldet_woodbury) < 1e-5


def test_ldet_sherman_woodbury_full():
    rng = np.random.default_rng(2021)
    p = 20
    L = 4
    A = rng.normal(size=(p, p))
    U = rng.normal(size=(p, L))
    B = rng.normal(size=(L, L))
    V = rng.normal(size=(L, p))
    UBV = np.dot(np.dot(U, B), V)
    AUBV = A + UBV
    _, ldet = la.slogdet(AUBV)
    ldet_woodbury = ldet_sherman_woodbury_full(A, U, B, V)
    assert np.abs(ldet - ldet_woodbury) < 1e-5


def test_get_conditional_parameters_sherman_woodbury():
    rng = np.random.default_rng(2021)
    W = rng.normal(size=(3, 10))
    Lambda = np.diag(np.abs(rng.normal(size=10)))
    covariance = Lambda + np.dot(W.T, W)
    idxs = np.ones(10)
    idxs[5:] = 0
    bb, cb = _get_conditional_parameters(covariance, idxs)
    bw, cw = _get_conditional_parameters_sherman_woodbury(Lambda, W, idxs)
    assert np.sum(np.abs(bb - bw)) < 1e-8
    assert np.sum(np.abs(cb - cw)) < 1e-8
