"""
rotation_adaptation.py

This module implements the RotationAdaptation class, which aims to 
harmonize data from multiple proteomics experiments that exhibit 
significant batch effects. It computes a centroid distribution and 
applies an affine transformation to each dataset, ensuring that all 
datasets align to the same mean and covariance as the centroid 
distribution. This alignment facilitates the combination and analysis 
of proteomics data from diverse sources by mitigating batch-specific 
differences.
"""
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import time


class RotationAdaptation:
    """
    A class for harmonizing multiple datasets by applying an affine
    transformation to align their means and covariances with those of
    a centroid distribution.

    Attributes
    ----------
    norm : str
        The norm to use for the optimization problem when computing the
        centroid covariance. Defaults to the Frobenius norm.

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

    def __init__(self, norm="fro"):
        """
        Initializes the RotationAdaptation instance with the specified norm.

        Parameters
        ----------
        norm : str, optional
            The norm to use for the optimization problem. Defaults to
            the Frobenius norm.
        """
        self.norm = norm

    def fit(self, X_list):
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
        self.J = len(X_list)

        mu_list = [np.mean(X_list[j], axis=0) for j in range(self.J)]
        Sigma_list = []
        L_i_cholesky_list = []
        for j in range(self.J):
            X_dm = X_list[j] - mu_list[j]
            cov = np.cov(X_dm.T) + 0.001 * np.eye(X_dm.shape[1])
            L = la.cholesky(cov)
            Sigma_list.append(cov)
            L_i_cholesky_list.append(la.inv(L))

        self.mu_0 = np.sum(mu_list, axis=0) / self.J

        Sigma_0_ = cp.Variable(Sigma_list[0].shape, symmetric=True)
        constraints = [Sigma_0_ >> 0]
        objective = cp.Minimize(
            cp.sum(
                [
                    cp.norm(Sigma_list[i] - Sigma_0_, self.norm)
                    for i in range(self.J)
                ]
            )
        )
        problem = cp.Problem(objective, constraints)
        startTime = time.time()
        problem.solve()
        self.elapsed_time = time.time() - startTime
        self.Sigma_0 = Sigma_0_.value
        self.L_0 = la.cholesky(self.Sigma_0)
        self.L_0_inv = la.inv(self.L_0)
        self.Sigma_list = Sigma_list
        self.L_i_cholesky = L_i_cholesky_list
        self.mu_list = mu_list

        return self

    def transform(self, X, j=None):
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
            Sigma = np.cov(X_dm.T)
            L = la.cholesky(Sigma)
            L_inv = la.inv(L)
            X_w = np.dot(X_dm, L_inv.T)
        else:
            X_dm = X - self.mu_list[j]
            X_w = np.dot(X_dm, self.L_i_cholesky[j].T)
        X_o = np.dot(X_w, self.L_0.T) + self.mu_0
        return X_o
