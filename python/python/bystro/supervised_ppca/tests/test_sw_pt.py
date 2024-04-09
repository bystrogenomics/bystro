import random
import torch
import numpy as np

from bystro.supervised_ppca._sherman_woodbury_pt import (
    ldet_sw_full,
    inverse_sw_full,
    ldet_sw_factor_analysis,
    inv_sw_factor_analysis,
    mvn_log_prob_sw,
)
from torch.distributions.multivariate_normal import MultivariateNormal

# Set global seeds
random.seed(0)
np.random.seed(0)  # noqa: NPY002


def test_ldet_sw_full():
    # Torch seeds do not work globally
    torch.manual_seed(0)

    # Test the log determinant computation
    A = torch.randn(3, 3) + torch.diag(10 * torch.rand(3)) + 1
    U = 0.1 * torch.randn(3, 3)
    B = 0.1 * torch.randn(3, 3)
    V = 0.1 * torch.randn(3, 3)

    result = ldet_sw_full(A, U, B, V)
    _, result_full = torch.slogdet(A + U @ B @ V)

    assert torch.is_tensor(result)
    assert torch.abs(result - result_full) < 1e-5


def test_inverse_sw_full():
    torch.manual_seed(0)

    # Test the inverse computation
    A = torch.randn(3, 3)
    U = torch.randn(3, 3)
    B = torch.randn(3, 3)
    V = torch.randn(3, 3)

    result = inverse_sw_full(A, U, B, V)
    res_true = torch.inverse(A + U @ B @ V)
    diff = result - res_true

    ss = torch.sum(torch.abs(diff))

    assert torch.is_tensor(result)
    assert ss < 1e-4


def test_ldet_sw_factor_analysis():
    torch.manual_seed(0)

    # Test the log determinant computation in factor analysis

    Lambda = torch.randn(3, 3) + torch.diag(10 * torch.rand(3)) + 1.0
    W = torch.randn(2, 3)

    result = ldet_sw_factor_analysis(Lambda, W)
    Sigma = Lambda + W.T @ W
    s1, res2 = torch.slogdet(Sigma)
    ss = torch.abs(result - res2)

    assert torch.is_tensor(result)
    assert ss < 1e-6


def test_inv_sw_factor_analysis():
    torch.manual_seed(0)

    # Test the inverse computation in factor analysis
    Lambda = torch.randn(3, 3) + torch.diag(10 * torch.rand(3))
    W = torch.randn(2, 3)

    result = inv_sw_factor_analysis(Lambda, W)
    Sigma = Lambda + W.T @ W
    res_true = torch.inverse(Sigma)
    diff = result - res_true
    ss = torch.sum(torch.abs(diff))

    assert torch.is_tensor(result)
    assert ss < 1e-4


def test_mvn_log_prob_sw():
    torch.manual_seed(0)

    # Test the log probability computation for multivariate normal
    p = 3
    X = torch.randn(5, p)
    mu = torch.randn(p)
    Lambda = torch.diag(torch.abs(torch.randn(p)))

    W = torch.randn(2, p)
    Sigma = Lambda + W.T @ W

    result = mvn_log_prob_sw(X, mu, Lambda, W)
    m = MultivariateNormal(mu, Sigma)

    like_tot = torch.mean(m.log_prob(X))

    assert torch.is_tensor(result)
    assert not torch.isnan(result).any()
    assert torch.abs(like_tot - result) < 1e-5
