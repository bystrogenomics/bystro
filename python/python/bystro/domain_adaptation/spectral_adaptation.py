"""
This module aligns the data using spectral features, with the goal of 
harmonizing data from multiple proteomics experiments that exhibit 
significant batch effects. There are several approaches implemented here,
all of which depend on aligning matrix quantities (such as spectral features
like eigenvalues) or means.

RotationAdaptation, known as CORAL, seeks to align the first and second 
moments of the different distributions to a centroid distribution. This 
may not be appropriate for very small sample sizes, so instead we allow for 
either just aligning the first moments (means), the variances, or a 
spectral alignment method. This avoids the concepts of estimating rotations,
and instead focuses on making sure that the eigenvalues of the covariance
matrices align. This exploits the fact that eigenvalues converge far 
quicker than the eigenvalues.

Objects
-------

Methods
-------
None
"""

from typing import List, Optional, Union

import numpy as np
import numpy.linalg as la

from bystro.covariance._covariance_np import (
    EmpiricalCovariance,
    LinearShrinkageCovariance,
    NonLinearShrinkageCovariance,
)
from bystro.covariance.covariance_cov_shrinkage import (
    GeometricInverseShrinkage,
    LinearInverseShrinkage,
    QuadraticInverseShrinkage,
)
from bystro.covariance.positive_definite_average import (
    pd_mean_harmonic,
    pd_mean_karcher,
    pd_mean_log_euclidean,
)


class MeanAdaptation:
    """
    A class to adapt datasets by aligning their means.

    This adaptation method works by calculating the mean of each dataset
    in a list of datasets (X_list) and adjusting each dataset to have
    the same mean. This can be useful in scenarios where datasets from
    different sources or conditions should be brought to a common scale
    or reference point before analysis.

    Parameters
    ----------
    prior_mu : bool, default=True
        If True, the adaptation process will include a regularization
        term towards zero in the calculation of the new means.
    lamb : float, default=0.1
        The regularization parameter used when `prior_mu` is True. It
        determines the weight of the regularization towards zero.
    weighted : bool, default=False
        If True, weights the contribution of each dataset by its number
        of samples in calculating the overall mean.

    Attributes
    ----------
    J : int
        The number of datasets.
    mu_list : List[np.ndarray]
        A list of means for each dataset in `X_list`.
    mu_0 : np.ndarray
        The target mean for all datasets after adaptation.
    n_covariates : int
        The number of features (covariates) in the datasets.
    """

    def __init__(self, prior_mu: bool = True, lamb: float = 0.1, weighted: bool = False) -> None:
        """
        Initializes the MeanAdaptation instance with the specified
        parameters

        Parameters
        ----------
        prior_mu : bool, default=True
            If True, the adaptation process will include a regularization
            term towards zero in the calculation of the new means.
        lamb : float, default=0.1
            The regularization parameter used when `prior_mu` is True. It
            determines the weight of the regularization towards zero.
        weighted : bool, default=False
            If True, weights the contribution of each dataset by its number
            of samples in calculating the overall mean.
        """
        self.prior_mu: bool = prior_mu
        self.lamb: float = lamb
        self.weighted: bool = weighted
        self.J: int = 0
        self.mu_list: List[np.ndarray] = []
        self.mu_0: np.ndarray = np.array([])
        self.n_covariates: int = 0

    def fit(self, X_list: List[np.ndarray]) -> "MeanAdaptation":
        """
        Fit the mean adaptation model on a list of datasets.

        Calculates the overall mean (optionally weighted by the size
        of each dataset) to which all datasets will be adapted.
        This mean can be adjusted towards zero if `prior_mu` is True.

        Parameters
        ----------
        X_list : List[np.ndarray]
            A list of numpy arrays where each array is a dataset to
            be adapted. Arrays must have the same number of columns
            (features).

        Returns
        -------
        self : MeanAdaptation
            The fitted MeanAdaptation instance.
        """
        self._test_inputs(X_list)
        self.J = len(X_list)
        mu_list = [np.mean(X, axis=0) for X in X_list]
        self.mu_list = mu_list

        if self.weighted:
            n_samples = np.array([X.shape[0] for X in X_list])
            weights = n_samples / np.sum(n_samples) * self.J
        else:
            weights = np.ones(len(mu_list))

        mu_avg = np.zeros(X_list[0].shape[1])
        for j, mu in enumerate(mu_list):
            mu_avg += weights[j] * mu
        mu_avg = mu_avg / self.J

        if self.prior_mu:
            self.mu_0 = mu_avg * (1 - self.lamb) + self.lamb * np.zeros(X_list[0].shape[1])
        else:
            self.mu_0 = mu_avg
        return self

    def transform(self, X: np.ndarray, j: Optional[int] = None) -> np.ndarray:
        """
        Transform a dataset to align its mean with the target mean.

        If `j` is specified, it aligns the dataset to the mean of the
        `j`th dataset from `X_list` used in fitting. Otherwise, it
        aligns to the overall target mean.

        Parameters
        ----------
        X : np.ndarray
            The dataset to transform, must have the same number of
            columns (features) as the datasets in `X_list`.
        j : int, optional
            The index of the dataset in `X_list` to align `X` to. If
            None, aligns to the overall target mean.

        Returns
        -------
        np.ndarray
            The dataset with its mean aligned to the target mean.
        """
        if X.shape[1] != self.n_covariates:
            raise ValueError("Mismatch in number of covariates")

        X_dm = X - np.mean(X, axis=0) if j is None else X - self.mu_list[j]
        X_o = X_dm + self.mu_0
        return X_o

    def _test_inputs(self, X_list: List[np.ndarray]) -> None:
        """
        Internal method to validate the input datasets.

        Checks that all datasets in `X_list` have the same number
        of features. Raises a ValueError if not.

        Parameters
        ----------
        X_list : List[np.ndarray]
            The list of datasets to validate.

        Raises
        ------
        ValueError
            If not all datasets in `X_list` have the same number of
            features.
        """
        self.n_covariates = X_list[0].shape[1]
        for X in X_list:
            if X.shape[1] != self.n_covariates:
                raise ValueError("Mismatch in number of covariates")


class RotationAdaptation:
    """
    A class for harmonizing multiple datasets by applying an affine
    transformation to align their means and covariances with those of
    a centroid distribution.

    Attributes
    ----------

    J : int
        The number of datasets.

    mu_0 : numpy.ndarray
        The mean of the centroid distribution.

    Sigma_0 : numpy.ndarray
        The covariance matrix of the centroid distribution.

    L_0_inv : numpy.ndarray
        The inverse of the Cholesky decomposition of the centroid
        covariance matrix.

    Sigma_list : list of numpy.ndarray
        The list of original covariance matrices for each dataset.

    L_i_cholesky : list of numpy.ndarray
        The list of inverses of the Cholesky decompositions of the covariance
        matrices for each dataset.

    mu_list : list of numpy.ndarray
        The list of means for each dataset.

    elapsed_time : float
        The time taken to solve the optimization problem for finding Sigma_0.

    Methods
    -------
    fit(X_list):
        Computes the centroid distribution and prepares the transformation
        parameters.

    transform(X, j=None):
        Applies the affine transformation to the given dataset X.
    """

    def __init__(
        self,
        centroid: str = "log_euclidean",
        regularization: str = "Linear",
        projection_regularization: str = "Linear",
    ) -> None:
        """
        Initializes the RotationAdaptation instance with the specified norm.

        Parameters
        ----------
        norm : str, optional
            The norm to use for the optimization problem. Defaults to
            the Frobenius norm.

        regularization: str,default="Linear",
            The regularization for estimating the covariance in each group

        projection_regularization: str,default="Linear",
            The regularization for estimating the covariance in the
            transform method with new group
        """
        self.J: int = 0
        self.centroid = centroid
        self.mu_0: np.ndarray = np.array([])
        self.Sigma_0: np.ndarray = np.array([])
        self.L_0_inv: np.ndarray = np.array([])
        self.Sigma_list: List[np.ndarray] = []
        self.L_i_cholesky: List[np.ndarray] = []
        self.mu_list: List[np.ndarray] = []
        self.elapsed_time: float = 0.0
        self.regularization = regularization
        self.projection_regularization = projection_regularization

    def fit(self, X_list: List[np.ndarray]) -> "RotationAdaptation":
        """
        Fits the model by computing the centroid mean and covariance
        matrix based on the provided list of datasets. This method
        also prepares the transformation parameters for each dataset.

        Parameters
        ----------
        X_list : list of numpy.ndarray
            A list containing the datasets to be harmonized. Each element
            of the list is a numpy array representing a dataset.
        """
        self._test_inputs(X_list)
        self.J = len(X_list)

        mu_list = [np.mean(X_list[j], axis=0) for j in range(self.J)]
        Sigma_list = []
        L_i_cholesky_list = []

        model_cov = _select_covariance_estimator(self.regularization)

        for j in range(self.J):
            X_dm = X_list[j] - mu_list[j]
            model_cov.fit(X_dm)
            if model_cov.covariance is None:
                raise ValueError("Covariance matrix is None.")
            cov = model_cov.covariance + 0.001 * np.eye(X_dm.shape[1])
            L = la.cholesky(cov)
            Sigma_list.append(cov)
            L_i_cholesky_list.append(la.inv(L))

        self.mu_0 = np.sum(mu_list, axis=0) / self.J

        if self.centroid == "log_euclidean":
            self.Sigma_0 = pd_mean_log_euclidean(Sigma_list)
        elif self.centroid == "harmonic":
            self.Sigma_0 = pd_mean_harmonic(Sigma_list)
        elif self.centroid == "karcher":
            self.Sigma_0 = pd_mean_karcher(Sigma_list)
        else:
            raise ValueError("Unrecognized option %s" % self.centroid)

        self.L_0 = la.cholesky(self.Sigma_0)
        self.L_0_inv = la.inv(self.L_0)
        self.Sigma_list = Sigma_list
        self.L_i_cholesky = L_i_cholesky_list
        self.mu_list = mu_list

        return self

    def transform(self, X: np.ndarray, j: Optional[int] = None) -> np.ndarray:
        """
        Projects the data into the new space. If the index is provided,
        corresponding to one of the input datasets, it uses that
        transformation to whiten the data. Otherwise,

        Parameters
        ----------
        X : np.ndarray
            The data to transform

        j : int, optional
            The index of which batch the data correspond to

        """
        if j is None:
            X_dm = X - np.mean(X, axis=0)
            model_cov = _select_covariance_estimator(self.projection_regularization)
            model_cov.fit(X_dm)
            Sigma = model_cov.covariance
            L = la.cholesky(Sigma)
            L_inv = la.inv(L)
            X_w = np.dot(X_dm, L_inv.T)
        else:
            X_dm = X - self.mu_list[j]
            X_w = np.dot(X_dm, self.L_i_cholesky[j].T)
        X_o = np.dot(X_w, self.L_0.T) + self.mu_0
        return X_o

    def _test_inputs(self, X_list: List[np.ndarray]) -> None:
        """
        Internal method to validate the input datasets.

        Checks that all datasets in `X_list` have the same number
        of features. Raises a ValueError if not.

        Parameters
        ----------
        X_list : List[np.ndarray]
            The list of datasets to validate.

        Raises
        ------
        ValueError
            If not all datasets in `X_list` have the same number of
            features.
        """
        self.n_covariates = X_list[0].shape[1]
        for X in X_list:
            if X.shape[1] != self.n_covariates:
                raise ValueError("Mismatch in number of covariates")


def _select_covariance_estimator(regularization):
    """
    This is a tiny method for selecting the estimator for the covariance
    matrix.
    """
    model_cov: Union[
        EmpiricalCovariance,
        LinearShrinkageCovariance,
        NonLinearShrinkageCovariance,
        LinearInverseShrinkage,
        GeometricInverseShrinkage,
        QuadraticInverseShrinkage,
    ]
    if regularization == "Empirical":
        model_cov = EmpiricalCovariance()
    elif regularization == "Linear":
        model_cov = LinearShrinkageCovariance()
    elif regularization == "LinearInverse":
        model_cov = LinearInverseShrinkage()
    elif regularization == "QuadraticInverse":
        model_cov = QuadraticInverseShrinkage()
    elif regularization == "GeometricInverse":
        model_cov = GeometricInverseShrinkage()
    elif regularization == "NonLinear":
        model_cov = NonLinearShrinkageCovariance()
    elif regularization == "Bayesian":
        raise ValueError("Bayesian currently not supported")
    else:
        raise ValueError("Unrecognized regularization %s" % regularization)
    return model_cov
