import numpy as np
from bystro.covariance.hypothesis_classical import (
    john_stat,
    john_sphericity_test,
    nagao_stat,
    nagao_identity_test,
    ledoit_wolf_stat,
    ledoit_wolf_identity_test,
    fisher_2012_stat_,
    fisher_single_sample_test,
    srivastava2011_,
    srivastava2011_one_sample_test,
    srivastava_yanagihara_stat,
    srivastavayanagihara_two_sample_test,
    srivastava_2007_stat,
    srivastava_two_sample_test,
)


def test_john_stat():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = john_stat(sample_data)
    assert isinstance(result, float)


def test_john_sphericity_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = john_sphericity_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result


def test_nagao_stat():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = nagao_stat(sample_data)
    assert isinstance(result, float)


def test_nagao_identity_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = nagao_identity_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result


def test_ledoit_wolf_stat():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = ledoit_wolf_stat(sample_data)
    assert isinstance(result, float)


def test_ledoit_wolf_identity_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = ledoit_wolf_identity_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result


def test_fisher_2012_stat_():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = fisher_2012_stat_(sample_data)
    assert isinstance(result, float)


def test_fisher_single_sample_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = fisher_single_sample_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result


def test_srivastava2011_():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = srivastava2011_(sample_data)
    assert isinstance(result, float)


def test_srivastava2011_one_sample_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n,p))
    result = srivastava2011_one_sample_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result


def test_srivastava_yanagihara_stat():
    rng = np.random.default_rng(2021)
    n1 = 50
    n2 = 60
    p = 5
    data1  = rng.normal(size=(n1,p))
    data2  = rng.normal(size=(n2,p))
    result = srivastava_yanagihara_stat([data1, data2])
    assert isinstance(result, float)


def test_srivastavayanagihara_two_sample_test():
    rng = np.random.default_rng(2021)
    n1 = 50
    n2 = 60
    p = 5
    data1  = rng.normal(size=(n1,p))
    data2  = rng.normal(size=(n2,p))
    result = srivastavayanagihara_two_sample_test([data1, data2])
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result


def test_srivastava_2007_stat():
    rng = np.random.default_rng(2021)
    n1 = 50
    n2 = 60
    p = 5
    data1  = rng.normal(size=(n1,p))
    data2  = rng.normal(size=(n2,p))
    result = srivastava_2007_stat([data1, data2])
    assert isinstance(result, float)


def test_srivastava_two_sample_test():
    rng = np.random.default_rng(2021)
    n1 = 50
    n2 = 60
    p = 5
    data1  = rng.normal(size=(n1,p))
    data2  = rng.normal(size=(n2,p))
    result = srivastava_two_sample_test([data1, data2])
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result
