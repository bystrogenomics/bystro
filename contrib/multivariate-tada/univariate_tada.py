import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import nbinom, gamma, poisson
import scipy.integrate as integrate
from tqdm import tqdm

# Based on original TADA software (He et al. 2013) and rewrite (Klei 2015)

# Set seed for reproducibility
np.random.seed(100)

def TADA(tada_counts : dict, sample_counts : dict, mu : pd.DataFrame, hyperpar : dict, denovo_only : pd.DataFrame, mu_frac : pd.DataFrame = 1, pi_gene : pd.DataFrame = 1):
    """
    Genome-wide application of TADA to obtain test statistics for K classes of variants

    Parameters
    ----------
    tada_counts : dict
        Dictionary of K data frames, each consisting of vectors for counts for denovo, case, and control mutation counts
    sample_counts : dict
        Dictionary of K data frames, each consisting of a vector with three entries of total sample counts,
        one for denovo (# trios), cases (# cases + # trios), and controls (# controls + # trios)
    mu : pd.DataFrame
        Dataframe with K vectors of mutation rates, one for each mutation category
    hyperpar : dict
        Dictionary of K data frames, each consisting of entries for gamma.mean.dn, beta.dn, gamma.mean.CC, beta.CC, rho1, nu1, rho0, nu0
    denovo_only : pd.DataFrame
        Dataframe with K Boolean variables indicating whether only denovo counts should be used (T)
        or whether both denovo and case-control counts be used (F)
    mu_frac : pd.DataFrame, optional
        Dataframe with the fraction to use for each mutation category K. Defaults to 1
    pi_gene : pd.DataFrame, optional
        Dataframe with K vectors of estimated fractions of causal variants, one for each class of variants.
        These fractions will be used to set gene-specific RR (case-control)

    Returns
    -------
        BF : pd.DataFrame
            A data frame with Bayes Factor (BF) for each of the K classes of variants as well as BF.total.
            One entry for each of the genes
    """


    # Ensures every list and dataframe has the same elements
    print("CALCULATION OF TADA TEST STATISTICS")
    print("checking the input for consistent variable names")
    mutation_types = list(tada_counts.keys())
    n_mutation = len(mutation_types)
    n_samples = len(list(tada_counts.values())[0])

    # Ensures mu_frac and pi_gene are data frames
    if not isinstance(mu_frac, pd.DataFrame):
        if isinstance(mu_frac,int):
            mu_frac = pd.DataFrame(np.reshape([[mu_frac] * n_mutation],(1,n_mutation)), columns=mutation_types)
        else:
            mu_frac = pd.DataFrame(mu_frac,columns=mutation_types)

    if not isinstance(pi_gene, pd.DataFrame):
        if isinstance(pi_gene,int):
            pi_gene = pd.DataFrame(np.reshape([[pi_gene] * n_mutation * n_samples],(n_samples,n_mutation,)), columns=mutation_types)
        else:
            pi_gene = pd.DataFrame(pi_gene,columns=mutation_types)

    if (
        sum([mutation_type in mu for mutation_type in mutation_types]) != n_mutation
        or sum([mutation_type in mu_frac for mutation_type in mutation_types]) != n_mutation
        or sum([mutation_type in hyperpar for mutation_type in mutation_types]) != n_mutation
        or sum([mutation_type in pi_gene for mutation_type in mutation_types]) != n_mutation
        or sum([mutation_type in denovo_only for mutation_type in mutation_types]) != n_mutation
        or sum([mutation_type in sample_counts for mutation_type in mutation_types]) != n_mutation
    ):
        return "mismatch in names for the different variables"

    names_N = ['dn', 'ca', 'cn']

    for mutation in mutation_types:
        if sum(names_N[i] in tada_counts[mutation] for i in range(3)) != 3:
            return f"columns of {mutation} do not match the required 'dn' 'ca' 'cn'"

    # Find the number of genes and the number of different kinds of mutations
    n_gene = len(mu)  # was m
    n_mutation = len(mutation_types)  # was K

    BF = pd.DataFrame()
    for mutation in mutation_types:
        print(f"working on :: {mutation}")
        BF_mut = np.array([])
        for i in range(len(tada_counts[mutation])):
            test_BF = calculate_BF(i,tada_counts[mutation], sample_counts[mutation], mu[mutation], mu_frac[mutation], hyperpar[mutation], denovo_only[mutation].item(), pi_gene[mutation])
            BF_mut = np.append(BF_mut,test_BF)

        BF = pd.concat([BF, pd.DataFrame(BF_mut)], axis=1)
    BF.columns = mutation_types
    BF.index = tada_counts[mutation].index

    # Calculate the overall BF
    BF_total = np.exp(np.log(BF).sum(axis=1))
    BF['BF.total'] = BF_total
    return BF

def TADAnull(tada_counts : list, sample_counts : list, mu : pd.Dataframe, hyperpar : pd.DataFrame, denovo_only : pd.DataFrame,
            mu_frac : pd.DataFrame = 1, n_rep : int = 100, dn_max : int = 20, ca_max : int = 200, cn_max : int = 200, max_gap : int = 50):
    """
    Genome-wide application of TADA for K classes of variants
    This function determines the distribution of the null hypothesis test statistics which in turn can be used to determine approximate p-values

    Parameters
    ----------
    tada_counts : list of pd.DataFrame
        List of K data frames, where each data frame consists of vectors for counts for denovo, case, and control mutation counts
    sample_counts : list of pd.DataFrame
        List of K data frames, where each data frame consists of a vector with three entries of total sample counts,
        one for denovo (# trios), cases (# cases + # trios), and controls (# controls + # trios)
    mu : pd.DataFrame
        Data frame with K vectors of mutation rates, one for each mutation category
    mu_frac : pd.DataFrame
        Data frame with the fraction to use for each mutation category K
    hyperpar : list of pd.DataFrame
        List of K data frames, where each data frame consists of entries for gamma.mean.dn, beta.dn, gamma.mean.CC, beta.CC, rho1, nu1, rho0, nu0
    denovo_only : pd.DataFrame
        Data frame with K Boolean variables indicating whether only denovo counts should be used (T) or whether both denovo and case-control counts should be used (F)
    n_rep : int
        Number of repetitions to use. Recommended to be at least 100. For smaller numbers of genes, n_rep should be increased
    dn_max : int
        Number of denovo events for which the BF is pre-computed and stored in a table. This speeds up the simulation process
        The function will use dn_max or the maximum number of denovo events for a gene, whichever is smaller.
    ca_max : int
        Maximum number of common case count events. Used to pre-compute a table of common case-control count events to speed up processing.
        Larger values result in longer pre-computation time. Integration error may occur for very large values
    cn_max : int
        Maximum number of common control count events. Used to pre-compute a table of common case-control count events to speed up processing.
        Larger values result in longer pre-computation time. Integration error may occur for very large values
    max_gap : int
        Gap between two genes in case-control count when ordered from smallest to largest count.
        Identifies outlying values that hardly ever occur and do not need to be pre-computed

    Returns
    -------
    dict
        Dictionary with BFnull for each of the K classes of variants as well as BFnull.total.
        One entry for each of the genes times n_rep
    """
    print("CALCULATION OF TADA TEST STATISTICS UNDER THE NULL HYPOTHESIS")
    mutation_types = list(tada_counts.keys())
    n_mutation = len(mutation_types)

    # Make sure mu_frac and pi_gene are data frames
    if not isinstance(mu_frac, pd.DataFrame):
        mu_frac = pd.DataFrame(np.reshape([[mu_frac] * n_mutation],(1,n_mutation)), columns=mutation_types)

    # Pre-compute the bayes factors for the denovo data
    table_BF_dn = {}
    for mutation in mutation_types:
        print(f"working on creating DN table for :: {mutation}")
        x = np.arange(dn_max + 1)
        param = hyperpar[mutation]
        n = sample_counts[mutation]['dn']
        BF = np.column_stack([bayes_factor_dn(x[i], n_dn=n.item(), mu=mu[mutation] * mu_frac[mutation].item(), gamma_dn=param["gamma.mean.dn"].item(), beta_dn=param["beta.dn"].item()) for i in range(len(x))])

        table_BF_dn[mutation] = pd.DataFrame(BF, columns=[f"X{value}" for value in x])
        tada_counts[mutation]["Ncc"] = tada_counts[mutation][["ca", "cn"]].sum(axis=1)

    table_BF_cc = {}
    for mutation in mutation_types:
        # Pre-compute the bayes factors for the case-control data
        if not denovo_only[mutation].item():
            print(f"working on creating CC table for :: {mutation}")
            tada_counts[mutation]["Ncc"] = tada_counts[mutation][["ca", "cn"]].sum(axis=1)
            Ncc = sorted(tada_counts[mutation]["Ncc"])
            Ncc_gaps = np.diff(Ncc)
            Ncc_max_gap = Ncc_gaps[np.where(Ncc_gaps > max_gap)[0][0]] if len(np.where(Ncc_gaps > max_gap)[0]) > 0 else max(ca_max,cn_max)
            n_ca = min(ca_max, Ncc_max_gap)
            n_cn = min(cn_max, Ncc_max_gap)
            x = pd.DataFrame(list(product(range(n_ca + 1), range(n_cn + 1))), columns=["ca", "cn"])
            param = hyperpar[mutation]
            n = sample_counts[mutation][["ca", "cn"]]
            BF = np.column_stack([bayes_factor_cc(x.loc[i], n_cc=n, gamma_cc=param["gamma.mean.CC"], beta_cc=param["beta.CC"], rho1=param["rho1"], nu1=param["nu1"], rho0=param["rho0"], nu0=param["nu0"]) for i in range(len(x))])
            BF = np.reshape(BF,(n_ca+1,n_cn+1))
            table_BF_cc[mutation] = pd.DataFrame(BF)

    # Determine BF under the null distribution through permutations
    BF = np.array([])
    for mutation in mutation_types:
        print("working on creating null data for ::", mutation)
        BF_col = [permute_gene(i,mu_rate=mu[mutation] * mu_frac[mutation][0], counts=tada_counts[mutation],
                                                        n=sample_counts[mutation], n_rep=n_rep, param=hyperpar[mutation],
                                                        denovo_only=denovo_only[mutation],
                                                        table_cc=table_BF_cc[mutation], table_dn=table_BF_dn[mutation]) for i in range(len(tada_counts[mutation]))]
        if(len(BF) != 0):
            BF = np.column_stack((BF,np.ravel(BF_col)))
        else:
            BF = np.append(BF,BF_col)

    # Calculate BF total
    BF = pd.DataFrame(BF, index = np.arange(len(BF)),columns=mutation_types)
    BF_total = np.exp(np.log(BF).sum(axis=1))

    return {'BF_null': BF, 'BF_null.total': BF_total}


def calculate_BF(i_gene : int, counts : pd.DataFrame, n : pd.DataFrame, mu : pd.Series, mu_frac : float, hyperpar : pd.DataFrame, denovo_only : bool, pi_gene : pd.Series):
    """ Wrapper function to calculate the BF for a gene and a particular mutation variant

    i_gene : int
        Gene of interest
    counts : pd.DataFrame
        Dataframe with vectors for dn, ca, and cn counts for a particular variant
    n : pd.DataFrame
        Dataframe with total sample counts of dn, ca, and cn
    mu : pd.Series
        Vector with mutation rates for the variant of interest for each gene
    mu_frac : float
        Fraction to multiply mu with for the variant of interest
    hyperpar : pd.DataFrame
        Dataframe with entries for gamma.mean.dn, beta.dn, gamma.mean.CC, beta.CC, rho1, nu1, rho0, nu0 for the
        variant of interest
    denovo_only : bool
        Boolean indicating whether only denovo contribution (denovo_only = True) or a combination of
        denovo and case-control contributions is to be used (denovo_only = False)
    pi_gene : pd.Series
        Vector with K vectors of estimated fractions of causal variants, one for each class of variants.
        These fractions will be used to set gene-specific RR (case-control)

    Returns
    -------
    float
        Bayes factor for the gene of interest
    """

    if i_gene % 100 == 0 or i_gene == len(counts)-1:
        pbar = tqdm(total=len(counts))
        pbar.update(100)

        if i_gene == len(counts)-1:
            pbar.close()

    # Set the hyperparameters for this gene
    hyperpar_gene = hyperpar.copy()
    RR_product = hyperpar_gene['gamma.mean.CC'] * hyperpar_gene['beta.CC']
    hyperpar_gene['gamma.mean.CC'] = hyperpar_gene['gamma.mean.CC'] * pi_gene[i_gene] + (1 - pi_gene[i_gene])
    hyperpar_gene['beta.CC'] = RR_product / hyperpar_gene['gamma.mean.CC']

    # Determine the Bayes factor
    BF = bayes_factor(x=counts.iloc[i_gene, :], n=n, mu=mu[i_gene] * mu_frac, param=hyperpar_gene, denovo_only=denovo_only)

    return BF

def bayes_factor(x : pd.Series, n : pd.DataFrame, mu : float, param : pd.Series, denovo_only : bool):
    """
    Calculate the Bayes Factor (BF) of the gene combining de novo and case-control data

    Parameters
    ----------
    x : pd.Series
        A vector of (dn, ca, cn) counts in de novo, cases, and controls for a given gene
    n : pd.DataFrame
        Dataframe with total sample counts of dn, ca, and cn
    mu : float
        Mutation rate of given gene
    param : pd.Series
        Parameters gamma.mean.dn, beta.dn, gamma.mean.CC, beta.CC, rho1, nu1, rho0, nu0 for prior distributions
    denovo_only : bool
        Boolean indicating whether only de novo contribution (True) or a combination of
        de novo and case-control contributions (False) is to be used

    Returns
    ------
    BF : float
        The calculated Bayes Factor
    """

    # Contribution of denovo variants in families
    BF_dn = bayes_factor_dn(x_dn=x['dn'], n_dn=n['dn'], mu=mu, gamma_dn=param['gamma.mean.dn'], beta_dn=param['beta.dn'])
    if not denovo_only:
        # Contribution of variants in cases and controls
        BF_cc = bayes_factor_cc(x_cc=x[['ca', 'cn']], n_cc=n[['ca', 'cn']], gamma_cc=param['gamma.mean.CC'], beta_cc=param['beta.CC'],
                                rho1=param['rho1'], nu1=param['nu1'], rho0=param['rho0'], nu0=param['nu0'])
    else:
        BF_cc = 1

    # Combine the pieces of information
    BF = BF_dn * BF_cc

    return BF

def bayes_factor_dn(x_dn : int, n_dn : int, mu : float, gamma_dn : float, beta_dn : float):
    """
    Calculates Bayes factor of de novo counts of a gene

    Parameters
    ----------
    x_dn : int
        De novo count
    n_dn : int
        Sample size (number of families)
    mu : float
        Mutation rate of this type of mutational event
    gamma_dn : float
        Parameter for prior distribution of relative risk
    beta_dn : float
        Parameter for prior distribution of relative risk

    Returns
    ------
    BF : float
        The calculated de novo Bayes Factor
    """

    # Prior distribution of RR: gamma ~ Gamma(gamma_dn * beta_dn, beta_dn)
    marg_lik0 = poisson.pmf(x_dn, 2 * n_dn * mu)
    marg_lik1 = nbinom.pmf(x_dn, gamma_dn * beta_dn, beta_dn / (beta_dn + 2 * n_dn * mu))
    BF = marg_lik1 / marg_lik0

    return BF


def bayes_factor_cc(x_cc : dict, n_cc : dict, gamma_cc : float, beta_cc : float, rho1 : float, nu1 : float, rho0 : float, nu0 : float):
    """
    Calculates Bayes factor of case-control data

    Parameters
    ----------
    x_cc : dict
        Case-control count data
    n_cc : dict
        Case-control sample sizes
    gamma_cc : float
        Parameter for prior distribution of relative risk
    beta_cc : float
        Parameter for prior distribution of relative risk
    rho1 : float
        Parameter for prior distribution of q|H1
    nu1 : float
        Parameter for prior distribution of q|H1
    rho0 : float
        Parameter for prior distribution of q|H0
    nu0 : float
        Parameter for prior distribution of q|H0

    Returns
    ------
    BF : float
        The calculated de novo Bayes Factor
    """

    # Prior distribution of RR: gamma ~ Gamma(gamma_cc * beta_cc, beta_cc)
    # Prior distribution of q|H1: Gamma(rho1, nu1)
    # Prior distribution of q|H0: Gamma(rho0, nu0)
    marglik0_cc = evidence_null_cc(x_cc, n_cc, rho0, nu0)
    marglik1_cc = evidence_alt_cc(x_cc, n_cc, gamma_cc, beta_cc, rho1, nu1)
    BF_cn = marglik1_cc["cn"] / marglik0_cc["cn"]
    BF_ca = marglik1_cc["ca"] / marglik0_cc["ca"]

    #Combines contributions from control and case data, respectively
    BF = BF_cn * BF_ca
    return BF

def evidence_null_cc(x_cc : dict, n_cc : dict, rho0 : float, nu0 : float):
    """
    Model evidence of case-control data: P(x_1,x_0|H_0)

    Parameters
    ----------
    x_cc : dict
        Case-control count data
    n_cc : dict
        Case-control sample sizes
    rho0 : float
        Parameter for prior distribution of q|H0
    nu0 : float
        Parameter for prior distribution of q|H0

    Returns
    ------
    dict
        The calculated evidence for null hypothesis
    """

    # Prior distribution of q|H0: Gamma(rho0, nu0)
    marglik0_ctrl_log = np.log(nbinom.pmf(x_cc["cn"], rho0, nu0 / (nu0 + n_cc["cn"])))
    marglik0_case_log = np.log(nbinom.pmf(x_cc["ca"], rho0 + x_cc["cn"], (nu0 + n_cc["cn"]) / (nu0 + n_cc["cn"] + n_cc["ca"])))
    marglik0_log = marglik0_ctrl_log + marglik0_case_log

    return {"cn": np.exp(marglik0_ctrl_log), "ca": np.exp(marglik0_case_log), "total": np.exp(marglik0_log)}

def evidence_alt_cc(x_cc : dict, n_cc : dict, gamma_cc : float, beta_cc : float, rho1 : float, nu1 : float, q_lower : float = 1e-8, q_upper : float = 0.1):
    """
    Model evidence of case-control data: P(x_1,x_0|H_1)

    Parameters
    ----------
    x_cc : dict
        Case-control count data
    n_cc : dict
        Case-control sample sizes
    gamma_cc : float
        Parameter for prior distribution of relative risk
    beta_cc : float
        Parameter for prior distribution of relative risk
    rho1 : float
        Parameter for prior distribution of q|H1
    nu1 : float
        Parameter for prior distribution of q|H1
    q_lower : float
        Lower bound of integration
    q_upper : float
        Upper bound of integration

    Returns
    ------
    dict
        The calculated evidence for alternative hypothesis
    """

    # Prior distribution of RR: gamma ~ Gamma(gamma_cc*beta_cc, bet a_cc)
    # Prior distribution of q|H1: Gamma(rho1, nu1)
    def integrand(u, x_ca, gamma_cc, beta_cc, n_ca, x_cn, rho1, nu1, n_cn):
        q = np.exp(u)
        return (nbinom.pmf(x_ca, gamma_cc * beta_cc, beta_cc / (beta_cc + n_ca * q)) *
            gamma.pdf(q, rho1 + x_cn, scale=1 / (nu1 + n_cn)) *
            np.exp(u))

    marglik1_ctrl = nbinom.pmf(x_cc["cn"], rho1, nu1 / (nu1 + n_cc["cn"]))
    marglik1_case = integrate.quad(integrand, np.log(q_lower), np.log(q_upper),args=(x_cc["ca"].item(), gamma_cc.item(), beta_cc.item(), n_cc["ca"].item(), x_cc["cn"].item(), rho1.item(), nu1.item(), n_cc["cn"].item()))[0]
    marglik1 = marglik1_ctrl * marglik1_case
    return {"cn": marglik1_ctrl, "ca": marglik1_case, "total": marglik1}

def permute_gene(i_gene : int, mu_rate : float, counts : pd.DataFrame, n : pd.DataFrame, n_rep : int, param : dict, denovo_only : bool, table_cc : pd.DataFrame, table_dn : pd.DataFrame):
    """
    Compute permutation Bayes factors (BFs) of one gene

    Parameters
    ----------
    i_gene : int
        Index of current gene
    mu_rate : float
        The mutation rate of a gene for the variant of interest
    counts : pd.DataFrame
        dn, ca, and cn counts for the variant of interest to be permuted, which also has a column for Ncc = ca + cn
    n : pd.DataFrame
        Sample size for de novo, case, control, and case+control
    n_rep : int
        Number of permutations
    param : dict
        Set of hyperparameters for the variant of interest
    denovo_only : bool
        Boolean indicating whether only de novo contribution (True) or a combination of
        de novo and case-control contributions (False) is to be used
    table_cc : pd.DataFrame
        Table of precomputed BFs for case-control events of size max.ca by max.cn for the variant of interest
    table_dn : pd.DataFrame
        Table of precomputed BFs for de novo events of size number of genes by max.dn for the variant of interest

    Returns
    -------
    BF : np.ndarray
        Array of n_rep BFs generated under the null hypothesis.
    """

    if i_gene % 100 == 0 or i_gene == counts.shape[0]:
        print(f"Progress: {i_gene}/{counts.shape[0]}")

    # Generate permutation data for denovo events
    sample_dn = np.random.poisson(2 * n['dn'].item() * mu_rate[i_gene], size=n_rep)
    # Look up the BF value in the table
    BF_dn = table_dn.iloc[i_gene].iloc[sample_dn]
    if not denovo_only.item():
        # When both denovo and case-control BF are needed
        # Generate permutation data for case-control events
        max_ca = table_cc.shape[0]
        max_cn = table_cc.shape[1]
        sample_ca = np.random.hypergeometric(counts["Ncc"][i_gene], n["ca"] + n["cn"] - counts["Ncc"][i_gene], n["ca"],n_rep)
        sample_cn = counts["Ncc"][i_gene] - sample_ca
        # Find the generated counts that are outside of the pre-computed table
        i_na = np.where((sample_ca + 1 > max_ca) | (sample_cn + 1 > max_cn))[0]

        if len(i_na) > 0:
            # Calculate their BF on a case by case basis
            BF_na = np.zeros(len(i_na))
            for idx, i in enumerate(i_na):
                BF_na[idx] = bayes_factor_cc({"ca": sample_ca[i], "cn": sample_cn[i]},n_cc=n[["ca", "cn"]], gamma_cc=param["gamma.mean.CC"], beta_cc=param["beta.CC"], rho1=param["rho1"], nu1=param["nu1"], rho0=param["rho0"], nu0=param["nu0"])

        # Gather the BF values that can be taken from the pre-computed table
        BF_cc = np.array([table_cc.iloc[sample_ca[i]][sample_cn[i]] if sample_ca[i] < max_ca and sample_cn[i] < max_cn else np.nan for i in range(n_rep) ])

        # Replace the missing values with the pre-computed ones
        i_na = np.where(np.isnan(BF_cc))[0]
        if len(i_na) > 0:
            BF_cc[i_na] = BF_na

    else:
        # If denovo only needed then set BF_dn to 1
        BF_cc = np.ones(n_rep)

    # Determine the total BF from the two components
    BF = BF_cc * BF_dn

    return BF


def Bayesian_FDR(BF : pd.Series, pi0 : float):
    """
    Bayesian FDR control

    Parameters
    ----------
    BF : pd.Series
        A vector of BFs
    pi0 : float
        The prior probability that the null model is true

    Returns
    -------
    FDR : list
        The q-value of each BF, and the number of findings with q below alpha.
    """

    # Order the BF in decreasing order, need to retain order to get results back in proper order
    i_order = np.argsort(-BF).to_numpy()
    BF = BF[i_order]

    # Convert BFs to PPA (posterior probability of alternative model)
    pi = 1 - pi0
    q = pi * BF / (1 - pi + pi * BF)  # PPA
    q0 = 1 - q  # posterior probability of null model

    # The FDR at each PPA cutoff
    FDR = np.cumsum(q0) / np.arange(1,len(BF)+1)

    # Reorder to the original order
    FDR = FDR[i_order]
    return FDR

def bayesFactor_pvalue(BF : pd.Series, BF_null : pd.Series):
    """
    Determines the p-value for the BF using permutation_types under the null hypothesis BF_null


    Parameters
    ----------
    BF : pd.Series
        Vector with bayes factors based on the data
    BF_null : pd.Series
        Vector with bayes factors based on permuted data

    Returns
    -------
    pval : pd.Series
        The p-values for the BFs
    """

    BF_null = np.sort(BF_null)[::-1]
    pval = np.searchsorted(-BF_null, -BF) / len(BF_null)
    pval[pval == 0] = 0.5 / len(BF_null)

    return pval

def denovo_MOM(k : int, N : int, mu : pd.DataFrame, C : int, beta : float, d : int = 2, S : int = 100, max_kvec : int = None):
    """
    Estimate relative risk and the number of multiple hits from de novo data

    Parameters
    ----------
    k : int
        Number of disease genes
    N : int
        Sample size
    mu : pd.DataFrame
        A table with mutation rate for every gene
    C : int
        Observed number of de novo events
    beta : float
        Parameter of the prior distribution of gamma
    d : int
        Number of events to use (1 is 1 or more, 2 is 2 or more)
    S : int
        Number of samples to generate per gene
    max_kvec : int
        Used to generate a timeline

    Returns
    -------
    dict
        Dictionary with gamma_mean representing average relative risk and M the expected number of multi-hit genes.
    """

    if max_kvec is not None:
        if k % 100 == 0 or k == max_kvec:
            pb = tqdm(total=max_kvec)
            pb.update(100)

    m = len(mu)  # Number of genes

    # Enrichment of de novo events
    nu = C / (2 * N * np.sum(mu))

    # MOM estimator of gamma_mean
    gamma_mean = (nu - 1) * m / k + 1

    # Expected M (choose d = 2)
    rs = count_multihit(N, mu, k / m, gamma_mean, beta, d=d, S=S)
    M = np.sum(rs['M1']) + np.sum(rs['M0'])

    return {'gamma.mean': gamma_mean, 'M': M}


def count_multihit(N : int, mu : pd.DataFrame, pi : float, gamma_mean : float, beta : float, d : int, S : int):
    """
    Estimate the number of multihit genes in a genome

    Parameters
    ----------
    N : int
        Sample size
    mu : pd.DataFrame
        A table with mutation rate for every gene
    pi : float
        Ratio of number of risk genes and total number of genes
    gamma_mean : float
        The average relative risk
    beta : float
        Parameter of the prior distribution of gamma
    d : int
        Number of events to use (1 is 1 or more, 2 is 2 or more)
    S : int
        Number of samples to generate per gene

    Returns
    -------
    dict
        Dictionary containing number of multihit genes for the non-risk genes and risk genes
    """

    m = len(mu)

    # M1: the number of causal genes having d or more de novo mutation_types
    p_alt = np.column_stack([multihit_prob(mu_i, N, gamma_mean, beta, d=d, S=S) for mu_i in mu])
    M1 = m * pi * np.mean(p_alt, axis=0)

    # M0: the number of non-causal genes having d or more de novo mutation_types
    p_null = np.column_stack([(1 - poisson.cdf(i, 2 * N * mu_i)) for mu_i in mu for i in range(d)])
    p_null = p_null.reshape(m, d)
    M0 = m * (1 - pi) * np.mean(p_null, axis=0)

    result = pd.DataFrame({'d': d, 'M0': M0, 'M1': M1})
    return result

def multihit_prob(mu : float, N : int, gamma_mean : float, beta : float, d : int, S : int):
    """
    Calculate the probability of having d or more de novo mutations under H1

    Parameters
    ----------
    mu : float
        Mutation rate for a gene
    N : int
        Sample size
    gamma_mean : float
        The average relative risk
    beta : float
        Parameter of the prior distribution of gamma
    d : int
        Number of events to use (1 is 1 or more, 2 is 2 or more)
    S : int
        Number of samples to generate per gene

    Returns
    -------
    float
        Average probability of having d or more de novo mutations.
    """

    gamma = gamma.rvs(gamma_mean * beta, scale=1 / beta, size=S)
    p = 1 - poisson.cdf(d, 2 * N * mu * gamma)
    return np.mean(p)