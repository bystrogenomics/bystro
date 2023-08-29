import numpy as np
from .._base_covariance import BaseCovariance


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
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    model.predict(X, idxs)
    


def test_conditional_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    model.conditional_score_samples(X, idxs)
    


def test_conditional_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    model.conditional_score(X, idxs)
    


def test_marginal_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    model.marginal_score(X[:, idxs == 1], idxs)
    


def test_marginal_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    model.marginal_score(X[:, idxs == 1], idxs, weights=0.5 * np.ones(1000))
    


def test_score():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    model.score(X, weights=0.5 * np.ones(1000))
    


def test_score_samples():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    model.score_samples(X)
    


def test_entropy():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    model.entropy()
    


def test_entropy_subset():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs = np.ones(10)
    idxs[8:] = 0
    model.entropy_subset(idxs)
    


def test_mutual_information():
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(1000, 10))
    model = BaseCovariance()
    model.covariance = 1 / X.shape[0] * np.dot(X.T, X)
    idxs1 = np.ones(10)
    idxs2 = np.ones(10)
    idxs1[5:] = 0
    idxs1 = np.ones(10)
    idxs2[:5] = 0
    idxs2[9] = 0
    model.mutual_information(idxs1, idxs2)
    
