import numpy as np
from bystro.covariance.optimal_shrinkage import (
    marcenko_pastur_integral,
    median_marcenko_pastur,
    inc_mar_pas,
    optimal_shrinkage,
)


def test_inc_mar_pas():
    assert np.abs(inc_mar_pas(1.0, 0.98, 0.8) - 0.68746165) < 1e-5
    assert np.abs(inc_mar_pas(1.0, 0.98, 0.7) - 0.63785129) < 1e-5
    assert np.abs(inc_mar_pas(0.9, 0.98, 0.7) - 0.66573444) < 1e-5
    assert np.abs(inc_mar_pas(0.2, 0.98, 0.7) - 0.84920378) < 1e-5
    assert np.abs(inc_mar_pas(0.2, 0.98, 0.7) - 0.84920378) < 1e-5
    assert np.abs(inc_mar_pas(0.2, 0.58, 0.7) - 0.90528871) < 1e-5


def test_median_marcenko_pastur():
    assert np.abs(median_marcenko_pastur(0.1) - 0.96653700) < 1e-4
    assert np.abs(median_marcenko_pastur(0.2) - 0.93293873) < 1e-4
    assert np.abs(median_marcenko_pastur(0.3) - 0.89910346) < 1e-4
    assert np.abs(median_marcenko_pastur(0.4) - 0.86482156) < 1e-4
    assert np.abs(median_marcenko_pastur(0.5) - 0.83052732) < 1e-4
    assert np.abs(median_marcenko_pastur(0.6) - 0.79552387) < 1e-4
    assert np.abs(median_marcenko_pastur(0.7) - 0.76080010) < 1e-4
    assert np.abs(median_marcenko_pastur(0.8) - 0.72520248) < 1e-4
    assert np.abs(median_marcenko_pastur(0.9) - 0.68959499) < 1e-4


def test_marcenko_pastur_integral():
    assert np.abs(marcenko_pastur_integral(1, 0.99) - 0.60841990) < 1e-4
    assert np.abs(marcenko_pastur_integral(1, 0.69) - 0.58974810) < 1e-4
    assert np.abs(marcenko_pastur_integral(1, 0.39) - 0.56692862) < 1e-4
    assert np.abs(marcenko_pastur_integral(1.8, 0.39) - 0.86127059) < 1e-4


def test_optimal_shrinkage():
    singular_values = np.array([1.50, 1.76, 41.31])

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "F_1")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 40.0854]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "F_2")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 31.0527]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "F_3")
    assert np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 6.8682]))) < 1e-3
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "F_4")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 40.6634]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "F_6")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 24.1599]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "N_1")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 39.4776]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "N_2")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 25.1051]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "N_3")
    assert np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 5.2411]))) < 1e-3
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "N_4")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 40.0561]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "N_6")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 16.8033]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "O_1")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 40.6931]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "O_2")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 40.6931]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "O_6")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 31.5164]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "Stein")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 31.0527]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "Ent")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 40.0854]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "Div")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 35.2811]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "Fre")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 39.7024]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2

    sigma_new, est_sigma = optimal_shrinkage(singular_values, 0.3, "Aff")
    assert (
        np.sum(np.abs(sigma_new - np.array([1.9575, 1.9575, 34.9621]))) < 1e-3
    )
    assert np.abs(est_sigma - 1.3991) < 1e-2
