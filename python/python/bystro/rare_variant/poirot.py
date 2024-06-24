"""
This module implements POIROT in Python, a statistical test to detect parent of
origin effects. Parent of origin effects occur when the impact of a mutation
depends on whether it was maternally or paternally inherited. POIROT works by
performing a Hotelling T2 test on the heterozygotes as opposed to the
homozygotes.

Functions implemented:
- group_center: Center data by group mean.
- center_pheno: Center phenotypes by genotype using mean or median.
- extract_residuals: Adjust for effects of covariates by extracting residuals.
- do_r_omnibus_test: Perform R-Omnibus test for equality of phenotypic covariance
  matrices.
- do_POIROT_by_snp: Perform POIROT test for one variant.


LICENSE FROM POIROT-POE

https://github.com/staylorhead/POIROT-POE/blob/main/LICENSE

Copyright (c) 2023 Taylor Head

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm  # type: ignore
from statsmodels.multivariate.manova import MANOVA  # type: ignore
from typing import List, Dict

pd.options.future.infer_string = True  # type: ignore


def group_center(data: pd.Series, group: pd.Series) -> pd.Series:
    """
    Center data by group mean.

    Parameters
    ----------
    data : pd.Series
        The data to be centered.
    group : pd.Series
        The group identifiers for each data point.

    Returns
    -------
    pd.Series
        The centered data.
    """
    return data.groupby(group).transform("mean")


def center_pheno(dat: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Center phenotypes by genotype.

    Parameters
    ----------
    dat : pd.DataFrame
        The input data frame containing phenotypes and genotype groups.
    by : str
        The method of centering. One of {'mean', 'median', 'none'}.

    Returns
    -------
    pd.DataFrame
        The data frame with centered phenotypes.
    """
    npheno = len([col for col in dat.columns if col.startswith("X")])

    if by == "mean":
        centered_vals = [group_center(dat[f"X{x+1}"], dat["off_geno_grp"]) for x in range(npheno)]
        centered_data = pd.DataFrame(
            np.array(centered_vals).T,
            columns=[f"pheno{x+1}" for x in range(npheno)],
        )
        dat = pd.concat([dat, centered_data], axis=1)

    elif by == "median":
        centered_vals = [
            dat.groupby("off_geno_grp")[f"X{x+1}"].transform("median") for x in range(npheno)
        ]
        dat[[f"pheno{x+1}" for x in range(npheno)]] = np.array(centered_vals).T

        cent_phenos = np.zeros((len(dat), npheno))
        for i in range(npheno):
            cent_phenos[dat["off_geno_grp"] == "AA", i] = (
                dat.loc[dat["off_geno_grp"] == "AA", f"X{i+1}"]
                - centered_vals[i][dat["off_geno_grp"] == "AA"]
            )
            cent_phenos[dat["off_geno_grp"] == "Aa", i] = (
                dat.loc[dat["off_geno_grp"] == "Aa", f"X{i+1}"]
                - centered_vals[i][dat["off_geno_grp"] == "Aa"]
            )
            cent_phenos[dat["off_geno_grp"] == "aa", i] = (
                dat.loc[dat["off_geno_grp"] == "aa", f"X{i+1}"]
                - centered_vals[i][dat["off_geno_grp"] == "aa"]
            )
        dat[[f"pheno{x+1}" for x in range(npheno)]] = cent_phenos

    elif by == "none":
        dat.rename(columns={f"X{x+1}": f"pheno{x+1}" for x in range(npheno)})

    return dat


def extract_residuals(phen_df: pd.DataFrame, covar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust for effects of covariates by extracting residuals.

    Parameters
    ----------
    phen_df : pd.DataFrame
        Data frame containing phenotype data.
    covar_df : pd.DataFrame
        Data frame containing covariate data.

    Returns
    -------
    pd.DataFrame
        Data frame containing residuals of the phenotype data after adjusting for covariates.
    """
    npheno = phen_df.shape[1]
    resid_mat = np.zeros((phen_df.shape[0], npheno))

    for pheno_ind in range(npheno):
        tmp = pd.concat([phen_df.iloc[:, pheno_ind], covar_df], axis=1)
        tmp.columns = pd.Index(["pheno"] + covar_df.columns.tolist())
        X = sm.add_constant(tmp.iloc[:, 1:])
        y = tmp["pheno"]
        model = sm.OLS(y, X).fit()
        resid_mat[:, pheno_ind] = model.resid

    resid_df = pd.DataFrame(resid_mat, columns=phen_df.columns)
    return resid_df


def do_r_omnibus_test(dat: pd.DataFrame, varnames: List[str], groupname: str) -> Dict[str, float]:
    """
    Perform R-Omnibus test for equality of phenotypic covariance matrices.

    Parameters
    ----------
    dat : pd.DataFrame
        The input data frame containing the data.
    varnames : List[str]
        The list of variable names to be tested.
    groupname : str
        The name of the grouping variable.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the p-value and test statistic.
    """
    dat = dat[[groupname] + varnames]
    npheno = len(varnames)
    grplevels = dat[groupname].unique()

    ngrp1 = (dat[groupname] == grplevels[0]).sum()
    ngrp2 = (dat[groupname] == grplevels[1]).sum()

    M = dat.groupby(groupname)[varnames].median().to_numpy()

    X = [
        dat[dat[groupname] == grplevels[0]][varnames].to_numpy(),
        dat[dat[groupname] == grplevels[1]][varnames].to_numpy(),
    ]

    x_M_1 = X[0] - M[0]
    x_M_2 = X[1] - M[1]

    Z = [
        [np.outer(x_M_1[j], x_M_1[j])[np.triu_indices(npheno)] for j in range(ngrp1)],
        [np.outer(x_M_2[j], x_M_2[j])[np.triu_indices(npheno)] for j in range(ngrp2)],
    ]

    W = []
    for group_Z in Z:
        group_W = []
        for z in group_Z:
            w = np.divide(z, np.sqrt(np.abs(z)), out=np.zeros_like(z), where=z != 0)
            group_W.append(w)
        W.append(group_W)

    W_df = pd.DataFrame({"grp": [grplevels[0]] * ngrp1 + [grplevels[1]] * ngrp2})
    W_df["grp"] = pd.Categorical(W_df["grp"])

    ntest = (npheno**2 + npheno) // 2
    mat_1 = np.array(W[0])
    mat_2 = np.array(W[1])
    mat = np.vstack([mat_1, mat_2])
    W_df = pd.concat([W_df, pd.DataFrame(mat)], axis=1)

    new_columns = ["grp"] + [f"test_{i}" for i in range(1, ntest + 1)]
    W_df.columns = pd.Index(new_columns)

    manova_model = MANOVA.from_formula(" + ".join(W_df.columns[1:]) + " ~ grp", data=W_df)
    result = manova_model.mv_test()

    pval = result.results["grp"]["stat"]["Pr > F"][0]
    stat = result.results["grp"]["stat"]["F Value"][0]

    return {"pval": pval, "stat": stat}


def do_poirot_by_snp(i: int, phenodat: pd.DataFrame, genodat: pd.DataFrame) -> Dict[str, float]:
    """
    Perform POIROT test for one variant.

    Parameters
    ----------
    i : int
        The index of the SNP to be tested.
    phenodat : pd.DataFrame
        Data frame containing phenotype data.
    genodat : pd.DataFrame
        Data frame containing genotype data.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the p-value and test statistic.
    """
    dat = phenodat.copy()
    dat["geno"] = genodat.iloc[:, i]
    dat["geno_grp"] = np.where(dat["geno"].isin([0, 2]), "Homozygote", "Heterozygote")
    dat["geno_grp"] = pd.Categorical(dat["geno_grp"], categories=["Homozygote", "Heterozygote"])
    npheno = phenodat.shape[1]

    medians = phenodat.groupby(dat["geno"]).median().to_numpy()

    centered_data = pd.DataFrame(
        phenodat.to_numpy() - medians[dat["geno"].to_numpy(), :],
        columns=[f"pheno{x+1}_cent" for x in range(npheno)],
    )
    dat = pd.concat([dat, centered_data], axis=1)

    result = do_r_omnibus_test(dat, [f"pheno{x+1}_cent" for x in range(npheno)], "geno_grp")
    return result
