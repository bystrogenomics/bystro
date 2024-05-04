import numpy as np
import pytest
from scipy.integrate import quad
from numerical_shrinkage import *

def test_numerical_shrinkage():
    eigenvalues = [2.1,3,8]
    gamma = 0.2
    loss = 'Stein'

    best_lam, best_loss, default_loss = numerical_shrinkage_impl(eigenvalues, gamma, loss)

    expected_best_lam = [1.0500, 2.0863, 6.3538]
    expected_best_loss = [0.0355, 0.0722, 0.0863]
    expected_default_loss = [0.1889, 0.1096, 0.1006]  
    
    assert np.allclose(best_lam, expected_best_lam)
    assert np.allclose(best_loss, expected_best_loss)
    assert np.allclose(default_loss, expected_default_loss)

def test_numerical_shrinkage_impl():
    lam = 3
    gamma = 0.5
    loss = 'Stein'
    best_lam, best_loss, default_loss = numerical_shrinkage_impl(lam, gamma, loss)
    expected_lam = 1.2000
    expected_best_loss = 0.0887
    expected_default_loss = 0.3806
    assert np.allclose(best_lam,expected_lam)
    assert np.allclose(best_loss,expected_best_loss)
    assert np.allclose(default_loss,expected_default_loss)

def test_bestLam_impl():
    A = np.array([[4, 1], [1, 2]])
    c = 0.6
    optLam, optVal = bestLam_impl(A, c, SteinLoss)
    expected_lam = 1.6509
    expected_val = 0.3480
    assert np.allclose(optLam,expected_lam)
    assert np.allclose(optVal,expected_val)
    
def test_defaultLoss_impl():
    lam = 1.465
    A = np.array([[4, 1], [1, 2]])
    c = 0.6
    risk = defaultLoss_impl(lam, A, c, SteinLoss)
    expected_risk = 0.3514
    assert np.allclose(risk,expected_risk)

def test_SteinLoss():
    A = np.array([[4, 1], [1, 2]])
    B = np.array([[3, 2], [2, 5]])
    J_value = SteinLoss(A, B)
    expected_J = 0.3454
    assert np.allclose(J_value,expected_J)