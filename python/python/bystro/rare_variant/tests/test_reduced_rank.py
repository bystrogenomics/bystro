import numpy as np
from bystro.rare_variant.reduced_rank import ReducedRankML

rng = np.random.default_rng(2021)
X = np.array(rng.normal(size=(100, 10)))
Y = np.array(rng.binomial(1, 0.5, size=(100, 5)))


def test_fit():
    model = ReducedRankML(lamb_sparsity=1.0, lamb_rank=1.0)
    model.fit(X, Y)

    # Check if the model has been trained
    assert hasattr(model, "B_") and model.B_ is not None
    assert hasattr(model, "alpha_") and model.alpha_ is not None


def test_predict():
    model = ReducedRankML(lamb_sparsity=1.0, lamb_rank=1.0)
    model.fit(X, Y)
    predictions = model.predict(X)

    # Check if the predictions have the correct shape
    assert predictions.shape == (X.shape[0], Y.shape[1])


def test_decision_function():
    model = ReducedRankML(lamb_sparsity=1.0, lamb_rank=1.0)
    model.fit(X, Y)
    scores = model.decision_function(X)

    # Check if the scores have the correct shape
    assert scores.shape == (X.shape[0], Y.shape[1])
