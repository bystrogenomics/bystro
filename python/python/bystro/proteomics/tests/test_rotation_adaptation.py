import numpy as np
import numpy.linalg as la
from scipy.stats import ortho_group
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from bystro.proteomics.rotation_adaptation import RotationAdaptation


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


def test_rotation_adaptation():
    n_clust = 3
    X_list = generate_data(n_clusters=n_clust)

    model = RotationAdaptation()
    model.fit(X_list)

    X_transformed_list = [model.transform(X_list[i], i) for i in range(n_clust)]
    cov_list = [np.cov(X_transformed_list[i].T) for i in range(n_clust)]

    for i in range(n_clust):
        assert la.norm(model.Sigma_0 - cov_list[i]) < 0.1
