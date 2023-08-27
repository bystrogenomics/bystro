import numpy as np
import numpy.linalg as la
from .._base_precision import BasePrecision


def test_get_stable_rank():
    model = BasePrecision()
    model.precision = np.eye(10)
    assert model.get_stable_rank() == 10


def test_predict():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    idxs = np.ones(10)
    idxs[8:] = 0
    model.predict(X, idxs)
    assert 1 == 1


def test_conditional_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    idxs = np.ones(10)
    idxs[8:] = 0
    model.conditional_score_samples(X, idxs)
    assert 1 == 1


def test_conditional_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    idxs = np.ones(10)
    idxs[8:] = 0
    model.conditional_score(X, idxs)
    assert 1 == 1


def test_marginal_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    idxs = np.ones(10)
    idxs[8:] = 0
    model.marginal_score(X[:, idxs == 1], idxs)
    assert 1 == 1


def test_marginal_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    idxs = np.ones(10)
    idxs[8:] = 0
    model.marginal_score(X[:, idxs == 1], idxs, weights=0.5 * np.ones(1000))
    assert 1 == 1


def test_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    model.score(X, weights=0.5 * np.ones(1000))
    assert 1 == 1


def test_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    model.score_samples(X)
    assert 1 == 1


def test_entropy():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    model.entropy()
    assert 1 == 1


def test_entropy_subset():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    idxs = np.ones(10)
    idxs[8:] = 0
    model.entropy_subset(idxs)
    assert 1 == 1


def test_mutual_information():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BasePrecision()
    model.precision = la.inv(1 / X.shape[0] * np.dot(X.T, X))
    idxs1 = np.ones(10)
    idxs2 = np.ones(10)
    idxs1[5:] = 0
    idxs1 = np.ones(10)
    idxs2[:5] = 0
    idxs2[9] = 0
    model.mutual_information(idxs1, idxs2)
    assert 1 == 1
