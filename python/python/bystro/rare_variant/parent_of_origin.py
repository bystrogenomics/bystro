"""
This module implements a parent of origin effect caller. The parent of 
origin effect refers to the difference in phenotypes 
depending on whether the allele is inherited paternally or maternally. The 
caller employs a method that draws inspiration from the community 
detection problem studied in computer science.

The distribution of phenotypes in the heterozygotes is modeled as a 
Gaussian mixture model. It assumes identical covariance matrices matching 
the homozygous population but with distinct means for the maternal and 
paternal populations. The model identifies this difference as the 
principal eigenvector of the heterozygote population, after undergoing a 
transformation that "whitens" the homozygous population. This process 
corresponds to finding the dimension along which the covariance is 
"stretched" compared to the heterozygote population.

A justification for this method can be found in Vershynin's "High-
Dimensional Probability" Chapter 4.7 or Wainwright's "High-Dimensional 
Statistics" Chapter 8.1.

Methods
-------
None

Objects
-------
BasePOE
    The base class of the parent of origin effect caller. Currently just 
    implements methods to test inputs for proper dimensionality and a
    method to classify heterozygotes based on their phenotypes

POESingleSNP(BasePOE)
    A caller for the POE of a single SNP. No sparsity assumptions are 
    implemented in this model.
"""
import numpy as np
import numpy.linalg as la
from typing import Tuple, Union, Optional
from tqdm import trange

from sklearn.utils import resample

from bystro.covariance.optimal_shrinkage import optimal_shrinkage
from bystro.covariance._covariance_np import (
    EmpiricalCovariance,
    NonLinearShrinkageCovariance,
)
from bystro.covariance.covariance_cov_shrinkage import (
    LinearInverseShrinkage,
    QuadraticInverseShrinkage,
)
from bystro.random_matrix_theory.rmt4ds_cov_test import two_sample_cov_test
from bystro.covariance.hypothesis_classical import (
    srivastavayanagihara_two_sample_test,
    srivastava_two_sample_test,
)


class BasePOE:
    """
    The base class of the parent of origin effect caller. Currently just
    implements methods to test inputs for proper dimensionality and a
    method to classify heterozygotes based on their phenotypes
    """

    def __init__(self) -> None:
        self.parent_effect_: np.ndarray = np.empty(
            10
        )  # Will be overwritten in fit

    def _test_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise ValueError("X is numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y is numpy array")
        if X.shape[0] != len(y):
            raise ValueError("X and y have different samples")

    def transform(
        self, X: np.ndarray, return_inner: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        This method predicts whether the heterozygote allele came from
        a maternal/paternal origin. Note that due to a lack of
        identifiability, we can't state whether class 1 is paternal or
        maternal

        Parameters
        ----------
        X : np.array-like, shape=(N, self.p)
            The phenotype data

        return_inner : bool, default=False
            Whether to return the inner product classification, a measure
            of confidence in the call

        Returns
        -------
        calls : np.array-like, shape=(N,)
            A vector of 1s and 0s predicting class

        preds : np.array-like, shape=(N,)
            The inner product, representing confidence in calls
        """
        X_dm = X - np.mean(X, axis=0)
        preds = np.dot(X_dm, self.parent_effect_)
        calls = 1.0 * (preds > 0)
        if return_inner is False:
            return calls
        return calls, preds


class POESingleSNP(BasePOE):
    """
    This is a parent of origin effect estimator inheriting methodology from
    the community detection problem commonly studied in computer science
    and statistics. It functions identically to a sklearn object, where
    model parameters are defined in the __init__ method, a fit method which
    takes in the data as input and fits the model, and a transform method
    which predicts which group new individuals belong to.

    Attributes
    ----------
    self.Sigma_AA : np.array-like, shape=(p,p)
        The covariance matrix of the homozygous population

    self.parent_effect_: np.array-like, shape=(p,)
        The difference in effect between the parental or maternal allele
    """

    def __init__(
        self,
        compute_pvalue: bool = False,
        compute_ci: bool = False,
        store_samples: bool = False,
        pval_method: str = "rmt4ds",
        n_permutations_pval: int = 10000,
        n_permutations_bootstrap: int = 10000,
        cov_regularization: str = "Empirical",
        svd_loss: Optional[str] = None,
    ) -> None:
        """
        Initialize the POESingleSNP estimator.

        Parameters
        ----------
        compute_pvalue : bool, optional, default=False
            Whether to compute p-values for the test.

        n_permutations : int, optional, default=10000
            The number of permutations to perform for significance testing.

        cov_regularization : str, optional, default="Empirical"
            The method of covariance regularization to use. Must be one of:
            'Empirical', 'NonLinear', 'LinearInverse', 'QuadraticInverse'.

        svd_loss : str or None, optional, default=None
            The type of SVD loss function to use. Should be a string specifying
            the loss function, or None if not applicable.

        Raises
        ------
        ValueError
            If `cov_regularization` is not one of the allowable values.
        """
        self.compute_pvalue = compute_pvalue
        self.compute_ci = compute_ci
        self.n_permutations_pval = n_permutations_pval
        self.n_permutations_bootstrap = n_permutations_bootstrap
        self.pval_method = pval_method
        self.store_samples = store_samples

        if cov_regularization == "Empirical":
            self.cov_reg: Union[
                EmpiricalCovariance,
                NonLinearShrinkageCovariance,
                LinearInverseShrinkage,
                QuadraticInverseShrinkage,
            ] = EmpiricalCovariance()
        elif cov_regularization == "NonLinear":
            self.cov_reg = NonLinearShrinkageCovariance()
        elif cov_regularization == "LinearInverse":
            self.cov_reg = LinearInverseShrinkage()
        elif cov_regularization == "QuadraticInverse":
            self.cov_reg = QuadraticInverseShrinkage()
        else:
            raise ValueError(
                "Invalid covariance regulator. Must be one of: Empirical, "
                "NonLinear, LinearInverse, QuadraticInverse"
            )
        self.svd_loss = svd_loss

        self.p_val = -1.0
        self.bootstrap_samples_: np.ndarray = np.empty(
            (10, 10)
        )  # Will be overwritten in fit
        self.confidence_interval_: np.ndarray = np.empty(
            (2, 10)
        )  # Will be overwritten in fit

    def fit(
        self, X: np.ndarray, y: np.ndarray, seed: int = 2021
    ) -> "POESingleSNP":
        """
        Fit the POESingleSNP model.

        Parameters
        ----------
        X : np.array-like, shape=(N, self.p)
            The phenotype data

        y : np.array-like, shape=(N,)
            The genotype data indicating the number of copies of the
            minority allele

        Returns
        -------
        self : POESingleSNP
            The instance of the method
        """
        self._test_inputs(X, y)
        self.n_phenotypes = X.shape[1]

        X_homozygotes = X[y == 0]
        X_heterozygotes = X[y == 1]
        X_homozygotes = X_homozygotes - np.mean(X_homozygotes, axis=0)
        X_heterozygotes = X_heterozygotes - np.mean(X_heterozygotes, axis=0)

        n_hetero = X_heterozygotes.shape[0]
        n_homo = X_homozygotes.shape[0]
        n_total = n_hetero + n_homo

        # Compute the point estimate
        self.cov_reg.fit(X_homozygotes)
        Sigma_AA = np.array(self.cov_reg.covariance)
        L = la.cholesky(Sigma_AA)
        L_inv = la.inv(L)

        self.Sigma_AA = Sigma_AA

        X_het_whitened = np.dot(X_heterozygotes, L_inv.T)
        Sigma_AB_white = np.cov(X_het_whitened.T)

        U, s, Vt = la.svd(Sigma_AB_white)

        if self.svd_loss:
            s, _ = optimal_shrinkage(
                s, self.n_phenotypes / n_hetero, self.svd_loss
            )

        norm_a = np.maximum(s[0] - 1, 0)
        parent_effect_white = Vt[0] * 2 * np.sqrt(norm_a)
        self.parent_effect_ = np.dot(parent_effect_white, L.T)

        # Compute the p value
        if self.compute_pvalue:
            rng = np.random.default_rng(seed)
            X_total = np.vstack((X_homozygotes, X_heterozygotes))

            if self.pval_method == "permutation":
                norms_p = np.zeros(self.n_permutations_pval)
                for i in trange(self.n_permutations_pval):
                    idx_hetero = np.zeros(n_total)
                    idx_hetero[
                        rng.choice(n_total, size=n_hetero, replace=False)
                    ] = 1
                    X_homo = X_total[idx_hetero == 0]
                    X_hetero = X_total[idx_hetero == 1]
                    X_homo = X_homo - np.mean(X_homo, axis=0)
                    X_hetero = X_hetero - np.mean(X_hetero, axis=0)
                    self.cov_reg.fit(X_homo)
                    Sigma_AA = np.array(self.cov_reg.covariance)
                    L = la.cholesky(Sigma_AA)
                    L_inv = la.inv(L)
                    X_het_whitened = np.dot(X_hetero, L_inv.T)
                    Sigma_AB_white = np.cov(X_het_whitened.T)

                    U, s, Vt = la.svd(Sigma_AB_white)
                    if self.svd_loss:
                        s, _ = optimal_shrinkage(
                            s,
                            self.n_phenotypes / n_hetero,
                            self.svd_loss,
                        )

                    norm_a = np.maximum(s[0] - 1, 0)
                    parent_effect_white = Vt[0] * 2 * np.sqrt(norm_a)
                    parent_effects = np.dot(parent_effect_white, L.T)
                    norms_p[i] = la.norm(parent_effects)

                self.p_val = float(
                    np.mean(norms_p > la.norm(self.parent_effect_))
                )
            elif self.pval_method == "rmt4ds":
                result = two_sample_cov_test(X_heterozygotes, X_homozygotes)
                self.p_val = result["p_value"]
            elif self.pval_method == "srivastavayanagihara":
                result = srivastavayanagihara_two_sample_test(
                    [X_heterozygotes, X_homozygotes]
                )
                self.p_val = result["p_value"]
            elif self.pval_method == "srivastava":
                result = srivastava_two_sample_test(
                    [X_heterozygotes, X_homozygotes]
                )
                self.p_val = result["p_value"]
            else:
                raise ValueError(
                    "Unrecognized p value option %s" % self.pval_method
                )

        # Bootstrap parameter confidence intervals
        if self.compute_ci:
            bootstrap_samples_ = np.zeros(
                (self.n_permutations_bootstrap, self.n_phenotypes)
            )
            for i in trange(self.n_permutations_bootstrap):
                X_homo = resample(X_homozygotes, n_samples=n_homo, replace=True)
                X_hetero = resample(
                    X_heterozygotes, n_samples=n_hetero, replace=True
                )
                X_homo = X_homo - np.mean(X_homo, axis=0)
                X_hetero = X_hetero - np.mean(X_hetero, axis=0)
                self.cov_reg.fit(X_homo)
                Sigma_AA = np.array(self.cov_reg.covariance)
                L = la.cholesky(Sigma_AA)
                L_inv = la.inv(L)
                X_het_whitened = np.dot(X_hetero, L_inv.T)
                Sigma_AB_white = np.cov(X_het_whitened.T)

                U, s, Vt = la.svd(Sigma_AB_white)
                if self.svd_loss:
                    s, _ = optimal_shrinkage(
                        s,
                        self.n_phenotypes / n_hetero,
                        self.svd_loss,
                    )

                norm_a = np.maximum(s[0] - 1, 0)
                parent_effect_white = Vt[0] * 2 * np.sqrt(norm_a)
                parent_effects = np.dot(parent_effect_white, L.T)
                if np.dot(parent_effects, self.parent_effect_) > 0:
                    bootstrap_samples_[i] = parent_effects
                else:
                    bootstrap_samples_[i] = -1 * parent_effects

            if self.store_samples:
                self.bootstrap_samples_ = bootstrap_samples_

            lb = np.quantile(bootstrap_samples_, 0.025, axis=0)
            ub = np.quantile(bootstrap_samples_, 0.975, axis=0)
            confidence_interval_ = np.zeros((2, self.n_phenotypes))
            confidence_interval_[0] = lb
            confidence_interval_[1] = ub

            self.confidence_interval_ = confidence_interval_
        return self
