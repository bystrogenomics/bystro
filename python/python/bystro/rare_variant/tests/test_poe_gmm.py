import numpy as np
from bystro.rare_variant.poe_gmm import POEGMM


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


def test_decision_function():
    rng = np.random.default_rng(2021)
    n_p = 10
    beta_m = np.zeros(n_p)
    beta_p = np.zeros(n_p)
    beta_p[:3] = 0.5
    data = generate_data(beta_m, beta_p, rng, maf=0.1, n_individuals=40000)
    model = POEGMM(
        mu=0.1, training_options={"n_iterations": 10000, "learning_rate": 1e-3}
    )
    model.fit(data["phenotypes"], data["genotype"],progress_bar=False)
    diff = beta_p - beta_m
    if model.parent_effect_ is None:
        raise ValueError("parent_effect_ is not initialized.")
    v1 = np.mean(np.abs(model.parent_effect_ - diff))
    v2 = np.mean(np.abs(model.parent_effect_ + diff))
    mse = np.minimum(v1, v2)
    assert mse < 0.2
