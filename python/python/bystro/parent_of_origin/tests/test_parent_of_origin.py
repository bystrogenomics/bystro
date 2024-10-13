"""
This code generates synthetic genotype and phenotype data for a genetic study.
It simulates individuals with or without a mother and/or a father, generates
genotypes based on minor allele frequency, and creates phenotype data based on
the genotype and parental status. The function returns a dictionary containing
various aspects of the generated data.
"""
import numpy as np
import numpy.linalg as la
from bystro.parent_of_origin.parent_of_origin import (
    POESingleSNP,
    POEMultipleSNP,
    POEMultipleSNP2,
)
import pytest


def cosine_similarity(vec1, vec2):
    v1 = np.squeeze(vec1)
    v2 = np.squeeze(vec2)
    num = np.dot(v1, v2)
    denom = la.norm(v1) * la.norm(v2)
    return num / denom


def test_min_sample_size_errors():
    # Simulate the case where there are fewer than 30 homozygotes and heterozygotes
    rng = np.random.default_rng(2021)

    # Set a very small number of homozygotes and heterozygotes (less than 30)
    n_homozygotes = 20
    n_heterozygotes = 20

    beta_m = np.zeros(10)  # Dummy beta values
    beta_p = np.zeros(10)
    beta_p[:3] = 0.5

    # Generate data with too few samples for both homozygotes and heterozygotes
    data = generate_data(
        beta_m, beta_p, rng, n_individuals=n_homozygotes + n_heterozygotes
    )

    # Instantiate the model
    model = POESingleSNP(
        compute_pvalue=True, cov_regularization="QuadraticInverse"
    )

    # Capture the exact error message manually, checking for the specific message
    with pytest.raises(ValueError) as excinfo:
        model.fit(data["phenotypes"], data["genotype"])

    # Ensure the message matches 'Too small of homozygous sample size, (>30)'
    assert "Too small of homozygous sample size, (>30)" in str(excinfo.value)


def generate_data(beta_m, beta_p, rng, n_individuals=10000, cov=None, maf=0.25):
    """
    Generate synthetic genotype and phenotype data for a genetic study.

    Parameters:
    -----------
    beta_m : array-like
        Coefficients for the mother's genotype.
    beta_p : array-like
        Coefficients for the father's genotype.
    rng : numpy.random.Generator
        Random number generator.
    n_individuals : int, optional
        Number of individuals in the study (default is 10000).
    cov : array-like, optional
        Covariance matrix for generating phenotype data (default is
        None, which corresponds to identity matrix).
    maf : float, optional
        Minor allele frequency for the genotypes (default is 0.25).

    Returns:
    --------
    dict:
        A dictionary containing the following keys:
        - has_mother: Array indicating whether each individual has a mother.
        - has_father: Array indicating whether each individual has a father.
        - genotype: Genotype of each individual.
        - n_indiv_neither: Number of individuals with neither mother nor father.
        - n_indiv_mother: Number of individuals with only a mother.
        - n_indiv_father: Number of individuals with only a father.
        - n_indiv_both: Number of individuals with both mother and father.
        - phenotypes: Phenotype data for all individuals.
    """

    results = {}
    p = len(beta_m)

    # If covariance matrix is not provided, use identity matrix
    if cov is None:
        cov = np.eye(p)

    # Generate binary indicators for having mother and father
    has_mother = rng.binomial(1, p=maf, size=n_individuals)
    has_father = rng.binomial(1, p=maf, size=n_individuals)
    genotype = has_mother + has_father

    # Store the generated data in the results dictionary
    results["has_mother"] = has_mother
    results["has_father"] = has_father
    results["genotype"] = genotype

    # Count the number of individuals in each category
    n_indiv_neither = np.sum((has_mother == 0) * (has_father == 0))
    n_indiv_mother = np.sum((has_mother == 1) * (has_father == 0))
    n_indiv_father = np.sum((has_mother == 0) * (has_father == 1))
    n_indiv_both = np.sum((has_mother == 1) * (has_father == 1))

    # Store the counts in the results dictionary
    results["n_indiv_neither"] = n_indiv_neither
    results["n_indiv_mother"] = n_indiv_mother
    results["n_indiv_father"] = n_indiv_father
    results["n_indiv_both"] = n_indiv_both

    # Generate phenotype data based on the genotype and parent status
    data_neither = rng.multivariate_normal(
        np.zeros(p), cov, size=n_indiv_neither
    )
    data_father = rng.multivariate_normal(beta_p, cov, size=n_indiv_father)
    data_mother = rng.multivariate_normal(beta_m, cov, size=n_indiv_mother)
    data_both = rng.multivariate_normal(beta_m + beta_p, cov, size=n_indiv_both)

    # Construct the phenotype matrix
    phenotypes = np.zeros((n_individuals, p))
    phenotypes[(has_mother == 0) & (has_father == 0)] = data_neither
    phenotypes[(has_mother == 1) & (has_father == 0)] = data_mother
    phenotypes[(has_mother == 0) & (has_father == 1)] = data_father
    phenotypes[(has_mother == 1) & (has_father == 1)] = data_both

    # Store the phenotype data in the results dictionary
    results["phenotypes"] = phenotypes

    return results


def generate_multivariate_data(
    beta_m,
    beta_p,
    rng,
    n_individuals=10000,
    cov=None,
    maf=0.25,
    n_genotypes=100,
):
    """
    Generate synthetic genotype and phenotype data for a genetic study. We are just going
    to use one relevant and the remainder are irrelevant

    Parameters:
    -----------
    beta_m : array-like
        Coefficients for the mother's genotype.
    beta_p : array-like
        Coefficients for the father's genotype.
    rng : numpy.random.Generator
        Random number generator.
    n_individuals : int, optional
        Number of individuals in the study (default is 10000).
    cov : array-like, optional
        Covariance matrix for generating phenotype data (default is
        None, which corresponds to identity matrix).
    maf : float, optional
        Minor allele frequency for the genotypes (default is 0.25).

    Returns:
    --------
    dict:
        A dictionary containing the following keys:
        - has_mother: Array indicating whether each individual has a mother.
        - has_father: Array indicating whether each individual has a father.
        - genotype: Genotype of each individual.
        - n_indiv_neither: Number of individuals with neither mother nor father.
        - n_indiv_mother: Number of individuals with only a mother.
        - n_indiv_father: Number of individuals with only a father.
        - n_indiv_both: Number of individuals with both mother and father.
        - phenotypes: Phenotype data for all individuals.
    """

    results = {}
    p = len(beta_m)

    # If covariance matrix is not provided, use identity matrix
    if cov is None:
        cov = np.eye(p)

    # Generate binary indicators for having mother and father
    has_mother = rng.binomial(1, p=maf, size=n_individuals)
    has_father = rng.binomial(1, p=maf, size=n_individuals)
    genotype = has_mother + has_father

    # Store the generated data in the results dictionary
    results["has_mother"] = has_mother
    results["has_father"] = has_father
    results["genotype"] = genotype

    # Count the number of individuals in each category
    n_indiv_neither = np.sum((has_mother == 0) * (has_father == 0))
    n_indiv_mother = np.sum((has_mother == 1) * (has_father == 0))
    n_indiv_father = np.sum((has_mother == 0) * (has_father == 1))
    n_indiv_both = np.sum((has_mother == 1) * (has_father == 1))

    # Store the counts in the results dictionary
    results["n_indiv_neither"] = n_indiv_neither
    results["n_indiv_mother"] = n_indiv_mother
    results["n_indiv_father"] = n_indiv_father
    results["n_indiv_both"] = n_indiv_both

    # Generate phenotype data based on the genotype and parent status
    data_neither = rng.multivariate_normal(
        np.zeros(p), cov, size=n_indiv_neither
    )
    data_father = rng.multivariate_normal(beta_p, cov, size=n_indiv_father)
    data_mother = rng.multivariate_normal(beta_m, cov, size=n_indiv_mother)
    data_both = rng.multivariate_normal(beta_m + beta_p, cov, size=n_indiv_both)

    # Construct the phenotype matrix
    phenotypes = np.zeros((n_individuals, p))
    phenotypes[(has_mother == 0) & (has_father == 0)] = data_neither
    phenotypes[(has_mother == 1) & (has_father == 0)] = data_mother
    phenotypes[(has_mother == 0) & (has_father == 1)] = data_father
    phenotypes[(has_mother == 1) & (has_father == 1)] = data_both

    # Store the phenotype data in the results dictionary
    results["phenotypes"] = phenotypes

    has_mother_total = rng.binomial(1, p=maf, size=(n_individuals, n_genotypes))
    has_father_total = rng.binomial(1, p=maf, size=(n_individuals, n_genotypes))
    has_mother_total[:, 0] = has_mother
    has_father_total[:, 0] = has_father
    genotype_total = has_mother_total + has_father_total
    results["genotypes"] = genotype_total

    return results


def test_decision_function():
    np.set_printoptions(suppress=True)
    rng = np.random.default_rng(2021)
    n_p = 10
    beta_m = np.zeros(n_p)
    beta_p = np.zeros(n_p)
    beta_p[:3] = 0.5
    data = generate_data(beta_m, beta_p, rng, maf=0.1, n_individuals=12000)
    model = POESingleSNP(
        compute_pvalue=True, cov_regularization="QuadraticInverse"
    )
    model.fit(data["phenotypes"], data["genotype"])
    diff = beta_p - beta_m
    assert np.abs(cosine_similarity(diff, model.parent_effect_)) > 0.95

    model = POESingleSNP(
        compute_pvalue=True,
        pval_method="permutation",
        compute_ci=True,
        cov_regularization="QuadraticInverse",
    )
    model.fit(data["phenotypes"], data["genotype"])
    assert model.p_val < 0.01


def test_multi_fit():
    np.set_printoptions(suppress=True)
    rng = np.random.default_rng(2021)
    n_p = 40
    beta_m = np.zeros(n_p)
    beta_p = np.zeros(n_p)
    beta_p[:3] = 0.5
    data = generate_multivariate_data(
        beta_m, beta_p, rng, maf=0.03, n_individuals=500, n_genotypes=100
    )
    model = POEMultipleSNP()
    model.fit(data["phenotypes"], data["genotypes"])


def test_multi2_fit():
    np.set_printoptions(suppress=True)
    rng = np.random.default_rng(2021)
    n_p = 40
    beta_m = np.zeros(n_p)
    beta_p = np.zeros(n_p)
    beta_p[:3] = 0.5
    data = generate_multivariate_data(
        beta_m, beta_p, rng, maf=0.03, n_individuals=1000, n_genotypes=100
    )
    model = POEMultipleSNP2(n_repeats=3)
    model.fit(data["phenotypes"], data["genotypes"], seed=2021)
    assert model is not None, "Model fitting failed"
    assert isinstance(model, POEMultipleSNP2), "Model type is incorrect"
