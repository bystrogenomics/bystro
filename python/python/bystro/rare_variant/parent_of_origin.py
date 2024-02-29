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


class BasePOE:
    """
    The base class of the parent of origin effect caller. Currently just
    implements methods to test inputs for proper dimensionality and a
    method to classify heterozygotes based on their phenotypes
    """
    def __init__(self):
        self.parent_effect_ = np.empty(10) # Will be overwritten in fit

    def _test_inputs(self, X, y):
        if not isinstance(X, np.ndarray):
            raise ValueError("X is numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y is numpy array")
        if X.shape[0] != len(y):
            raise ValueError("X and y have different samples")

    def transform(self, X, return_inner=False):
        """
        This method predicts whether the heterozygote allele came from
        a maternal/paternal origin. Note that due to a lack of
        identifiability, we can't state whether class 1 is paternal or
        maternal

        Parameters
        ----------
        X : np.array-like,shape=(N,self.p)
            The phenotype data

        return_inner : bool,default=False
            Whether to return the inner product classification, a measure
            of confidence in the call

        Returns
        -------
        calls : np.array-like,shape=(N,)
            A vector of 1s and 0s predicting class

        preds : np.array-like,shape=(N,)
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
    the commumity detection problem commonly studied in computer science
    and statistics. It functions identically to a sklearn object, where
    model parameters are defined in the __init__ method, a fit method which
    takes in the data as input and fits the model, and a transform method
    which predicts which group new individuals belong to.

    Attributes
    ----------
    self.Sigma_AA : np.array-like,shape=(p,p)
        The covariance matrix of the homozygous population

    self.parent_effect_: np.array-like,shape=(p,)
        The difference in effect between the parental or maternal allele
    """

    def __init__(self, compute_pvalue=False, n_permutations=10000):
        self.compute_pvalue = compute_pvalue
        self.n_permutations = n_permutations

    def fit(self, X, y):
        """
        This method predicts whether the heterozygote allele came from
        a maternal/paternal origin. Note that due to a lack of
        identifiability, we can't state whether class 1 is paternal or
        maternal

        Parameters
        ----------
        X : np.array-like,shape=(N,self.p)
            The phenotype data

        y: np.array-like,shape=(N,)
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
        X_homozygotes = X_homozygotes - np.mean(X_homozygotes,axis=0)
        X_heterozygotes = X_heterozygotes - np.mean(X_heterozygotes,axis=0)

        Sigma_AA = np.cov(X_homozygotes.T)
        L = la.cholesky(Sigma_AA)
        L_inv = la.inv(L)

        self.Sigma_AA = Sigma_AA

        X_het_whitened = np.dot(X_heterozygotes, L_inv.T)
        Sigma_AB_white = np.cov(X_het_whitened.T)

        U,s,Vt = la.svd(Sigma_AB_white)
        norm_a = np.maximum(s[0]-1,0)
        parent_effect_white = Vt[0]*2*np.sqrt(norm_a)
        self.parent_effect_ = np.dot(parent_effect_white,L.T)
        return self
