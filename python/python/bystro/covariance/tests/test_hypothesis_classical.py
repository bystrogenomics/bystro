import numpy as np
from bystro.covariance.hypothesis_classical import (
    john_stat,
    john_sphericity_test,
    nagao_stat,
    nagao_identity_test,
    ledoit_wolf_stat,
    ledoit_wolf_identity_test,
    fisher_single_sample_test,
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
    sample_data = rng.normal(size=(n, p))
    result = john_stat(sample_data)
    assert isinstance(result, float)


def test_john_sphericity_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n, p))
    result = john_sphericity_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result

    p = 10
    n = 50
    N = 10000

    alt_cov_X = rng.random((p, p))
    alt_cov_X = alt_cov_X @ alt_cov_X.T
    alt_cov_X += np.eye(p)

    X_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    X_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    X_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_X, N)

    assert john_sphericity_test(X_small_null)["p_value"] > 0.05
    assert john_sphericity_test(X_large_null)["p_value"] > 0.05
    assert john_sphericity_test(X_large_alt)["p_value"] < 0.001


def test_nagao_stat():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n, p))
    result = nagao_stat(sample_data)
    assert isinstance(result, float)


def test_nagao_identity_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n, p))
    result = nagao_identity_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result

    p = 10
    n = 50
    N = 10000

    alt_cov_X = rng.random((p, p))
    alt_cov_X = alt_cov_X @ alt_cov_X.T
    alt_cov_X += np.eye(p)

    X_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    X_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    X_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_X, N)

    assert nagao_identity_test(X_small_null)["p_value"] > 0.05
    assert nagao_identity_test(X_large_null)["p_value"] > 0.05
    assert nagao_identity_test(X_large_alt)["p_value"] < 0.001


def test_ledoit_wolf_stat():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n, p))
    result = ledoit_wolf_stat(sample_data)
    assert isinstance(result, float)


def test_ledoit_wolf_identity_test():
    rng = np.random.default_rng(2021)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n, p))
    result = ledoit_wolf_identity_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result

    p = 10
    n = 50
    N = 10000

    alt_cov_X = rng.random((p, p))
    alt_cov_X = alt_cov_X @ alt_cov_X.T
    alt_cov_X += np.eye(p)

    X_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    X_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    X_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_X, N)

    assert ledoit_wolf_identity_test(X_small_null)["p_value"] > 0.05
    assert ledoit_wolf_identity_test(X_large_null)["p_value"] > 0.05
    assert ledoit_wolf_identity_test(X_large_alt)["p_value"] < 0.001


def test_fisher_single_sample_test():
    rng = np.random.default_rng(2022)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n, p))
    result = fisher_single_sample_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result

    p = 10
    n = 50
    N = 10000

    alt_cov_X = rng.random((p, p))
    alt_cov_X = alt_cov_X @ alt_cov_X.T
    alt_cov_X += np.eye(p)

    X_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    X_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    X_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_X, N)

    print("_________________________________")
    print(fisher_single_sample_test(X_small_null))
    print(fisher_single_sample_test(X_large_null))
    print(fisher_single_sample_test(X_large_alt))
    print("_________________________________")

    assert fisher_single_sample_test(X_small_null)["p_value"] > 0.05
    assert fisher_single_sample_test(X_large_null)["p_value"] > 0.05
    assert fisher_single_sample_test(X_large_alt)["p_value"] < 0.001


def test_srivastava2011_one_sample_test():
    rng = np.random.default_rng(2022)
    n = 50
    p = 5
    sample_data = rng.normal(size=(n, p))
    result = srivastava2011_one_sample_test(sample_data)
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result

    p = 10
    n = 50
    N = 10000

    alt_cov_X = rng.random((p, p))
    alt_cov_X = alt_cov_X @ alt_cov_X.T
    alt_cov_X += np.eye(p)

    X_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    X_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    X_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_X, N)

    assert srivastava2011_one_sample_test(X_small_null)["p_value"] > 0.05
    assert srivastava2011_one_sample_test(X_large_null)["p_value"] > 0.05
    assert srivastava2011_one_sample_test(X_large_alt)["p_value"] < 0.001


def test_srivastava_yanagihara_stat():
    rng = np.random.default_rng(2021)
    n1 = 50
    n2 = 60
    p = 5
    data1 = rng.normal(size=(n1, p))
    data2 = rng.normal(size=(n2, p))
    result = srivastava_yanagihara_stat([data1, data2])
    assert isinstance(result, float)


def test_srivastavayanagihara_two_sample_test():
    rng = np.random.default_rng(2021)
    n1 = 50
    n2 = 60
    p = 5
    data1 = rng.normal(size=(n1, p))
    data2 = rng.normal(size=(n2, p))
    result = srivastavayanagihara_two_sample_test([data1, data2])
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result

    p = 5
    n = 50
    N = 20000

    alt_cov_X = rng.random((p, p))
    alt_cov_X = alt_cov_X @ alt_cov_X.T
    alt_cov_X += np.eye(p)
    alt_cov_Y = rng.random((p, p))
    alt_cov_Y = alt_cov_Y @ alt_cov_Y.T
    alt_cov_Y += np.eye(p)

    X_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    Y_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    X_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    Y_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    X_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_X, N)
    Y_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_Y, N)

    assert (
        srivastavayanagihara_two_sample_test([X_small_null, Y_small_null])[
            "p_value"
        ]
        > 0.05
    )
    assert (
        srivastavayanagihara_two_sample_test([X_large_null, Y_large_null])[
            "p_value"
        ]
        > 0.05
    )
    assert (
        srivastavayanagihara_two_sample_test([X_large_alt, Y_large_alt])[
            "p_value"
        ]
        < 0.001
    )


def test_srivastava_2007_stat():
    rng = np.random.default_rng(2021)
    n1 = 50
    n2 = 60
    p = 5
    data1 = rng.normal(size=(n1, p))
    data2 = rng.normal(size=(n2, p))
    result = srivastava_2007_stat([data1, data2])
    assert isinstance(result, float)


def test_srivastava_two_sample_test():
    rng = np.random.default_rng(2022)
    n1 = 50
    n2 = 60
    p = 5
    data1 = rng.normal(size=(n1, p))
    data2 = rng.normal(size=(n2, p))
    result = srivastava_two_sample_test([data1, data2])
    assert isinstance(result, dict)
    assert "stat" in result
    assert "p_value" in result

    p = 5
    n = 50
    N = 10000

    alt_cov_X = rng.random((p, p))
    alt_cov_X = alt_cov_X @ alt_cov_X.T
    alt_cov_X += np.eye(p)
    alt_cov_Y = rng.random((p, p))
    alt_cov_Y = alt_cov_Y @ alt_cov_Y.T
    alt_cov_Y += np.eye(p)

    X_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    Y_small_null = rng.multivariate_normal(np.zeros(p), np.eye(p), n)
    X_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    Y_large_null = rng.multivariate_normal(np.zeros(p), np.eye(p), N)
    X_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_X, N)
    Y_large_alt = rng.multivariate_normal(np.zeros(p), alt_cov_Y, N)

    assert (
        srivastava_two_sample_test([X_small_null, Y_small_null])["p_value"]
        > 0.05
    )
    assert (
        srivastava_two_sample_test([X_large_null, Y_large_null])["p_value"]
        > 0.05
    )
    assert (
        srivastava_two_sample_test([X_large_alt, Y_large_alt])["p_value"]
        < 0.001
    )
