import pickle
import os

import numpy as np

from bystro.covariance.hypothesis_hd import (
    sy2010,
    clx2013,
    hc2018,
    two_sample_test,
    hd2017,
    schott2007,
)

def get_pickle_file_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "hypothesis_hd_data.pkl")


def test_sy2010():
    with open(get_pickle_file_path(), "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    Y = data["Y"]

    result = sy2010(X, Y)
    expected = {"Q2": 0.9652861, "pvalue": 0.3258586}

    assert np.isclose(
        result["Q2"], expected["Q2"]
    ), f"Expected {expected['Q2']}, got {result['Q2']}"
    assert np.isclose(
        result["pvalue"], expected["pvalue"]
    ), f"Expected {expected['pvalue']}, got {result['pvalue']}"


def test_clx2013():
    with open(get_pickle_file_path(), "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    Y = data["Y"]

    result = clx2013(X, Y)
    expected = {"TSvalue": -3.567752, "pvalue": 0.6949956}

    assert np.isclose(
        result["TSvalue"], expected["TSvalue"]
    ), f"Expected {expected['TSvalue']}, got {result['TSvalue']}"
    assert np.isclose(
        result["pvalue"], expected["pvalue"]
    ), f"Expected {expected['pvalue']}, got {result['pvalue']}"


def test_hc2018():
    with open(get_pickle_file_path(), "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    Y = data["Y"]

    result = hc2018(X, Y)
    expected_reject = 0
    expected_pvalues = [
        0.27137703,
        0.58025294,
        0.46391632,
        0.80002848,
        0.35552749,
        0.88377427,
        0.33080928,
        0.51855715,
        0.96743237,
        0.76494794,
        0.49859619,
        0.29214629,
        0.33176071,
        0.59950532,
        0.68317829,
        0.64311724,
        0.17317045,
        0.60601858,
        0.60477422,
        0.85456787,
        0.19327118,
        0.07969246,
        0.85893915,
        0.37388840,
        0.15964912,
        0.59500536,
        0.30049437,
        0.64120179,
        0.12031435,
        0.79068842,
        0.27979323,
        0.39937620,
        0.97308226,
        0.97093054,
        0.90724718,
        0.36761294,
        0.09712775,
        0.19536379,
        0.49927272,
        0.14753675,
        0.45265371,
    ]
    expected_N = 40

    assert (
        result["reject"] == expected_reject
    ), f"Expected {expected_reject}, got {result['reject']}"
    assert len(result["pvalues"]) == len(
        expected_pvalues
    ), f"Expected {len(expected_pvalues)} p-values, got {len(result['pvalues'])}"
    for res_pval, exp_pval in zip(result["pvalues"], expected_pvalues):
        assert np.isclose(
            res_pval, exp_pval, rtol=1e-6
        ), f"Expected {exp_pval}, got {res_pval}"
    assert (
        result["N"] == expected_N
    ), f"Expected {expected_N}, got {result['N']}"


def test_two_sample_test():
    with open(get_pickle_file_path(), "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    Y = data["Y"]

    result = two_sample_test(X, Y, seed=2021)
    expected = {
        "decision": 0,
        "statistic": 83.34365038789211,
        "pvalue": 0.8854185908161136,
        "df": 100,
        "reject": 4,
    }

    assert (
        result["decision"] == expected["decision"]
    ), f"Expected decision {expected['decision']}, got {result['decision']}"
    assert np.isclose(
        result["statistic"], expected["statistic"]
    ), f"Expected statistic {expected['statistic']}, got {result['statistic']}"
    assert np.isclose(
        result["pvalue"], expected["pvalue"]
    ), f"Expected pvalue {expected['pvalue']}, got {result['pvalue']}"
    assert (
        result["df"] == expected["df"]
    ), f"Expected df {expected['df']}, got {result['df']}"
    assert (
        result["reject"] == expected["reject"]
    ), f"Expected reject {expected['reject']}, got {result['reject']}"


def test_hd2017():
    with open(get_pickle_file_path(), "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    Y = data["Y"]

    result = hd2017(X, Y, J=2500, seed=2021)
    expected = {"statistics": 3.9948, "p.value": 0.5924}

    assert np.isclose(
        result["statistics"], expected["statistics"]
    ), f"Expected {expected['statistics']}, got {result['statistics']}"
    assert np.isclose(
        result["p.value"], expected["p.value"], rtol=1e-3
    ), f"Expected {expected['p.value']}, got {result['p.value']}"


def test_schott2007():
    with open(get_pickle_file_path(), "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    Y = data["Y"]

    result = schott2007(X, Y)
    expected = {"statistics": 0.82999, "p.value": 0.4065}

    assert np.isclose(
        result["statistics"], expected["statistics"]
    ), f"Expected {expected['statistics']}, got {result['statistics']}"
    assert np.isclose(
        result["p.value"], expected["p.value"], rtol=1e-3
    ), f"Expected {expected['p.value']}, got {result['p.value']}"
