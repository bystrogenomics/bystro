import numpy as np
from scipy.io import loadmat  # type: ignore
from bystro.covariance.rmt_cov_est import rmt_estim, rmt_estim_rgrad, rmtest
from typing import Tuple, Optional


# Generate test matrices function with np.random.Generator
def generate_test_matrices(
    p: int, n: int, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random matrices for testing.

    Parameters
    ----------
    p : int
        Dimension of the square matrix.
    n : int
        Number of samples.
    seed : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - x: A random matrix of size (p, n).
        - C0: An initial guess for the covariance matrix (p, p).
        - C: True covariance matrix of size (p, p).
    """
    rng = np.random.default_rng(seed)  # Create a random generator with a seed
    x = rng.standard_normal((p, n))
    C0 = np.eye(p)
    C = np.cov(x)
    return x, C0, C


# Function to extract MATLAB result fields (MATLAB stores them as structured arrays)
def extract_matlab_results(
    matlab_struct: dict, distance: str, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    method_results = matlab_struct[distance][0][0][method][0][0]
    if method == "rmt_estim":
        out = method_results["out"][0]
        r = method_results["r"][0]
    elif method == "rmt_estim_rgrad":
        out = method_results["out_grad"][0]
        r = method_results["grad"][0]
    else:
        raise ValueError(f"Unknown method: {method}")
    return out, r


# Test functions


def test_rmt_estim():
    # Load the test data
    matlab_results = loadmat("test_matrices.mat")
    C0 = np.array(matlab_results["C0"])
    C = np.array(matlab_results["C"])
    n = int(matlab_results["n"])

    distance_metrics = [
        "log",
        "log1st",
        "t",
        "KL",
        "Battacharrya",
        "Inverse_log",
        "Inverse_log1st",
        "Inverse_t",
        "Inverse_KL",
        "Inverse_Battacharrya",
    ]

    for distance in distance_metrics:
        # Extract MATLAB results
        out_matlab, r_matlab = extract_matlab_results(
            matlab_results, distance, "rmt_estim"
        )

        # Python results
        out_python, r_python = rmt_estim(C0, C, n, distance)

        # Compare results
        np.testing.assert_array_almost_equal(out_python, out_matlab, decimal=6)
        np.testing.assert_array_almost_equal(r_python, r_matlab, decimal=6)


def test_rmt_estim_rgrad():
    # Load the test data
    matlab_results = loadmat("test_matrices.mat")
    C0 = np.array(matlab_results["C0"])
    C = np.array(matlab_results["C"])
    n = int(matlab_results["n"])

    distance_metrics = [
        "Fisher",
        "log",
        "log1st",
        "t",
        "KL",
        "Battacharrya",
        "Inverse_Fisher",
        "Inverse_log",
        "Inverse_log1st",
        "Inverse_t",
        "Inverse_KL",
        "Inverse_Battacharrya",
    ]

    for distance in distance_metrics:
        # Extract MATLAB results
        out_grad_matlab, grad_matlab = extract_matlab_results(
            matlab_results, distance, "rmt_estim_rgrad"
        )

        # Python results
        out_grad_python, grad_python = rmt_estim_rgrad(C0, C, n, distance)

        # Compare results
        np.testing.assert_array_almost_equal(
            out_grad_python, out_grad_matlab, decimal=6
        )
        np.testing.assert_array_almost_equal(
            grad_python, grad_matlab, decimal=6
        )


def test_rmtest():
    # Load the test data
    matlab_results = loadmat("test_matrices.mat")
    x = np.array(matlab_results["x"])
    C0 = np.array(matlab_results["C0"])
    C = np.array(matlab_results["C"])

    distance_metrics = [
        "Fisher",
        "log",
        "log1st",
        "t",
        "KL",
        "Battacharrya",
        "Inverse_Fisher",
        "Inverse_log",
        "Inverse_log1st",
        "Inverse_t",
        "Inverse_KL",
        "Inverse_Battacharrya",
    ]

    for distance in distance_metrics:
        # Python results
        C_est_python, cost_python = rmtest(
            x, C0, C, check_gradient=False, plot_cost=False, distance=distance
        )

        # TODO: Compare with MATLAB results when they are available
