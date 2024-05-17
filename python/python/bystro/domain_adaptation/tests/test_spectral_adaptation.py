import numpy as np
import numpy.linalg as la
from scipy.stats import ortho_group  # type: ignore
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pytest

from bystro.domain_adaptation.spectral_adaptation import (
    MeanAdaptation,
    RotationAdaptation,
)


def generate_data(n_clusters=3, N=10000, p=1000, seed=2021):
    rng = np.random.default_rng(2021)
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    data_standardized = StandardScaler().fit_transform(data)
    U, S, VT = la.svd(data_standardized, full_matrices=False)

    X_list = []
    for i in range(n_clusters):
        s_new = np.zeros(p)
        s_new[:64] = np.maximum(0, S + rng.normal(size=64) * 20)
        VT_new = ortho_group.rvs(p, random_state=seed + i)
        U_new = np.zeros((data.shape[0], p))
        U_new[:, :64] = U
        d_new = U_new @ np.diag(s_new) @ VT_new
        scaled_new = StandardScaler().fit_transform(d_new)
        cov_i = np.cov(scaled_new.T) + 100.0 * np.eye(p)
        mu_i = rng.normal(size=p) + i
        X_list.append(rng.multivariate_normal(mu_i, cov_i, size=N))

    return X_list


def test_mean_adaptation():
    # Create a Generator object
    rng = np.random.default_rng(seed=42)

    # Generate three datasets with different means
    X1 = rng.normal(loc=0, scale=1, size=(100, 10))
    X2 = rng.normal(loc=5, scale=1, size=(100, 10))
    X3 = rng.normal(loc=10, scale=1, size=(100, 10))
    X_list = [X1, X2, X3]

    # Instantiate and fit MeanAdaptation
    mean_adapt = MeanAdaptation()
    mean_adapt.fit(X_list)

    # Transform the datasets
    X1_transformed = mean_adapt.transform(X1)
    X2_transformed = mean_adapt.transform(X2)
    X3_transformed = mean_adapt.transform(X3)

    # Verify that the transformed datasets have means close to the common mean
    common_mean = mean_adapt.mu_0
    assert np.allclose(
        np.mean(X1_transformed, axis=0), common_mean, atol=0.1
    ), "X1 not correctly adapted."
    assert np.allclose(
        np.mean(X2_transformed, axis=0), common_mean, atol=0.1
    ), "X2 not correctly adapted."
    assert np.allclose(
        np.mean(X3_transformed, axis=0), common_mean, atol=0.1
    ), "X3 not correctly adapted."

    with pytest.raises(ValueError):
        X4 = rng.normal(
            loc=15, scale=1, size=(100, 5)
        )  # Different number of features
        mean_adapt.fit([X1, X4])


def test_rotation_adaptation():
    n_clust = 3
    X_list = generate_data(n_clusters=n_clust)

    model = RotationAdaptation(regularization="Empirical")
    model.fit(X_list)

    X_transformed_list = [model.transform(X_list[i], i) for i in range(n_clust)]
    cov_list = [np.cov(X_transformed_list[i].T) for i in range(n_clust)]

    for i in range(n_clust):
        assert (
            la.norm(model.Sigma_0 - cov_list[i]) / la.norm(cov_list[i]) < 0.01
        )
