import numpy as np
import pytest
from scipy.integrate import quad
from optimal_shrinkage import *

# Test incMarPas function
def test_incMarPas():
    x0 = 0.5
    gamma = 0.9
    alpha = 1.5
    result = incMarPas(x0, gamma, alpha)
    assert result == 1.2830201689652752

# Test MedianMarcenkoPastur function
def test_MedianMarcenkoPastur():
    gamma = 0.9
    result = MedianMarcenkoPastur(gamma)
    assert result == 0.6895949913056676
    
# Test MarcenkoPasturIntegral function
def test_MarcenkoPasturIntegral():
    x = 1.5
    gamma = 0.7
    result = MarcenkoPasturIntegral(x, gamma)
    assert result ==  0.737909026656867

# Test optimal_shrinkage function
def test_optimal_shrinkage():
    eigenvals = np.array([0.5, 1.0, 1.5])
    gamma = 0.9
    loss = "F_1"
    sigma = 0.1
    result, _ = optimal_shrinkage(eigenvals, gamma, loss, sigma)
    assert result == np.array([0.48179444, 0.98189916, 1.48193319])

# Test ell function
def test_ell():
    lam = np.array([0.5, 1.0, 1.5])
    lam_plus = 1.0
    gamma = 0.5
    result = ell(lam, lam_plus, gamma)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(lam)

# Test c function
def test_c():
    lam = np.array([0.5, 1.0, 1.5])
    lam_plus = 1.0
    gamma = 0.5
    result = c(lam, lam_plus, gamma)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(lam)

# Test s function
def test_s():
    lam = np.array([0.5, 1.0, 1.5])
    lam_plus = 1.0
    gamma = 0.5
    result = s(lam, lam_plus, gamma)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(lam)

if __name__ == "__main__":
    pytest.main([__file__])
