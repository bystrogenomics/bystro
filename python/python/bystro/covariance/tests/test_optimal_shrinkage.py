import numpy as np
import pytest
from bystro.covariance.optimal_shrinkage import *


def test_inc_mar_pas():
    x0 = 0.5
    gamma = 0.9
    alpha = 1.5
    expected_result = 1.2830
    assert np.isclose(inc_mar_pas(x0, gamma, alpha), expected_result, 1e-4)

    expected_result = 1.7435
    assert np.isclose(inc_mar_pas(0.7, 0.8, 2), expected_result, 1e-4)

    with pytest.raises(ValueError):
        inc_mar_pas(0.9, 0, 3)

    with pytest.raises(ValueError):
        inc_mar_pas(0.5, 1.1, 1.5)


def test_median_marcenko_pastur():
    gamma = 0.9
    expected_result = 0.6896
    assert np.isclose(median_marcenko_pastur(gamma), expected_result, atol=1e-4)

    expected_result = 0.8305
    assert np.isclose(median_marcenko_pastur(0.5), expected_result, atol=1e-4)

    with pytest.raises(ValueError):
        median_marcenko_pastur(1.1)


def test_marcenko_pastur_integral():
    x = 1.5
    gamma = 0.7
    expected_result = 0.7379
    assert np.isclose(marcenko_pastur_integral(x, gamma), expected_result, atol=1e-4)

    with pytest.raises(ValueError):
        marcenko_pastur_integral(2, 1.1)
    with pytest.raises(ValueError):
        marcenko_pastur_integral(2, 0)
    with pytest.raises(ValueError):
        marcenko_pastur_integral(2, 0.1)


def test_optimal_shrinkage():
    eigenvals = np.array([0.5, 1.0, 1.5])
    gamma = 0.9
    loss = "F_1"
    sigma = 0.1
    expected_result = np.array([0.4818, 0.9819, 1.4819])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, loss, sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.2581, 0.5213, 0.7845])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "F_2", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.0202, 0.0207, 0.0208])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "F_3", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.4906, 0.9908, 1.4909])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "F_4", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.1404, 0.279, 0.4175])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "F_6", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.4728, 0.9729, 1.4729])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "N_1", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.1751, 0.3537, 0.5322])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "N_2", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.0106, 0.0109, 0.0109])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "N_3", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.4816, 0.9818, 1.4819])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "N_4", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.0181, 0.0319, 0.0458])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "N_6", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.4908, 0.9909, 1.4909])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "O_1", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.4908, 0.9909, 1.4909])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "O_2", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.2628, 0.526, 0.7892])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "O_6", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.2581, 0.5213, 0.7845])
    assert np.allclose(
        optimal_shrinkage(eigenvals, gamma, "Stein", sigma)[0], expected_result, atol=1e-4
    )

    expected_result = np.array([0.4818, 0.9819, 1.4819])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "Ent", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.3526, 0.7154, 1.0782])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "Div", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.4752, 0.9746, 1.4743])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "Fre", sigma)[0], expected_result, atol=1e-4)

    expected_result = np.array([0.3352, 0.6801, 1.0249])
    assert np.allclose(optimal_shrinkage(eigenvals, gamma, "Aff", sigma)[0], expected_result, atol=1e-4)
