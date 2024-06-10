"""
Matrix Operations with Sherman-Woodbury Identity

This module provides functions for matrix operations utilizing the 
Sherman-Woodbury identity. It includes computations for log determinants, 
inverses, and log probabilities, specifically tailored for factor 
analysis and matrix products. The Sherman-Woodbury identity allows for 
efficient calculations when dealing with inverses of low rank +  diagonal, 
which is necessary for evaluating normal likelihoods and converting between
precision/covariance.

Objects
-------
None

Methods
-------
- `ldet_sw_full`: Compute the log determinant of a matrix product using the Sherman-Woodbury identity.
- `inverse_sw_full`: Compute the inverse of a matrix product using the Sherman-Woodbury identity.
- `ldet_sw_factor_analysis`: Compute the log determinant of the covariance matrix in a factor 
  analysis model using the Sherman-Woodbury identity.
- `inv_sw_factor_analysis`: Compute the inverse of the covariance matrix in a factor analysis model.
- `mvn_log_prob_sw`: Compute the log probability of data points under a multivariate normal 
  distribution using the Sherman-Woodbury matrix identity.

Parameters Used
---------------
A : torch.Tensor
    Input matrices used in the calculations.
U : torch.Tensor
    Input matrices used in the calculations.
B : torch.Tensor
    Input matrices used in the calculations.
V : torch.Tensor
    Input matrices used in the calculations.
Lambda : torch.Tensor
    Diagonal matrix in factor analysis.
W : torch.Tensor
    Factor loading matrix.
I_l : torch.Tensor, optional
    LxL identity matrix. If not provided, it is set to the identity matrix.
I_p : torch.Tensor, optional
    PxP identity matrix. If not provided, it is set to the identity matrix.

References
----------
Sherman, J., & Morrison, W. J. (1950). Adjustment of an Inverse Matrix 
     Corresponding to Changes in   the Elements of a Given Column or a 
     Given Row of the Original Matrix. The Annals of Mathematical 
     Statistics, 21(1), 124–127.

Woodbury, M. A. (1950). Inverting Modified Matrices. Memorandum Report, 
     42, 1–12.
"""
import torch
import torch.linalg as la


def ldet_sw_full(A, U, B, V):
    """
    Compute the log determinant of a matrix product:

    M = (A + UBV)

    Parameters
    ----------
    A : torch.Tensor
        Input matrix A.

    U : torch.Tensor
        Input matrix U.

    B : torch.Tensor
        Input matrix B.

    V : torch.Tensor
        Input matrix V.

    Returns
    -------
    torch.Tensor
        Log determinant of the matrix product.
    """
    # Compute the log determinant of matrix A
    s1, ldet_A = torch.slogdet(A)

    # Compute the log determinant of matrix B
    s2, ldet_B = torch.slogdet(B)

    # Compute the second term involving matrix V, A, and U
    term2 = torch.matmul(V, la.solve(A, U))

    # Compute the inverse of matrix B
    term1 = la.inv(B)

    # Compute the log determinant of the matrix product
    s3, ldet_prod = torch.slogdet(term1 + term2)

    # Compute the final log determinant
    log_determinant = ldet_A + ldet_B + ldet_prod

    return log_determinant


def inverse_sw_full(A, U, B, V):
    """
    Compute the inverse of a matrix product.

    M = (A + UBV)

    Parameters
    ----------
    A : torch.Tensor
        Input matrix A.

    U : torch.Tensor
        Input matrix U.

    B : torch.Tensor
        Input matrix B.

    V : torch.Tensor
        Input matrix V.

    Returns
    -------
    torch.Tensor
        Inverse of the matrix product.
    """
    # Compute the inverse of matrix A
    A_inv = la.inv(A)

    # Compute the inverse of matrix B
    B_inv = la.inv(B)

    # Compute the product A_inv * U
    AiU = torch.matmul(A_inv, U)

    # Compute the product V * A_inv
    VAinv = torch.matmul(V, A_inv)

    # Compute the product V * A_inv * U
    VAiU = torch.matmul(VAinv, U)

    # Compute the sum of B_inv and V * A_inv * U
    middle = B_inv + VAiU

    # Solve the system of linear equations using the product V * A_inv
    end = la.solve(middle, VAinv)

    # Compute the product A_inv * U * (solved system)
    second_term = torch.matmul(AiU, end)

    # Compute the inverse of the matrix product
    Sigma_inv = A_inv - second_term

    return Sigma_inv


def ldet_sw_factor_analysis(Lambda, W, I_l=None):
    """
    Compute the log determinant of the covariance matrix in a factor
    analysis model using the Sherman Woodbury identity

    Parameters
    ----------
    Lambda : torch.Tensor
        Diagonal component of covariance matrix

    W : torch.Tensor
        Factor loading matrix.

    I_l : torch.Tensor, optional
        LxL identity matrix. If not provided, it is set to the identity matrix.

    Returns
    -------
    torch.Tensor
        Log determinant of the precision matrix.
    """
    if I_l is None:
        I_l = torch.eye(W.shape[0])
    ldet_L = torch.logdet(Lambda)
    LiW = la.solve(Lambda, torch.transpose(W, 0, 1))
    WtLiW = torch.matmul(W, LiW)
    IWtLiW = I_l + WtLiW
    ldet_p = torch.logdet(IWtLiW)
    log_determinant = ldet_p + ldet_L
    return log_determinant


def inv_sw_factor_analysis(Lambda, W, I_l=None, I_p=None):
    """
    Compute the inverse of the covariance matrix in a factor analysis model.

    Parameters
    ----------
    Lambda : torch.Tensor
        Diagonal component of covariance matrix

    W : torch.Tensor
        Factor loading matrix.

    I_l : torch.Tensor, optional
        LxL identity matrix. If not provided, it is set to the identity matrix.

    I_p : torch.Tensor, optional
        PxP identity matrix. If not provided, it is set to the identity matrix.

    Returns
    -------
    torch.Tensor
        Inverse of the covariance matrix.
    """
    if I_l is None:
        I_l = torch.eye(W.shape[0])
    if I_p is None:
        I_p = torch.eye(W.shape[1])

    Lambda_inv = torch.inverse(Lambda)
    WLi = torch.matmul(W, Lambda_inv)
    inner = I_l + torch.matmul(WLi, torch.transpose(W, 0, 1))
    inner_inv = la.inv(inner)
    end = torch.matmul(inner_inv, WLi)
    term2 = torch.matmul(torch.transpose(W, 0, 1), end)
    Imterm2 = I_p - term2
    Sigma_inv = torch.matmul(Lambda_inv, Imterm2)
    return Sigma_inv


def mvn_log_prob_sw(X, mu, Lambda, W, I_l=None):
    """
    Compute the log probability of data points under a multivariate
    normal distribution using the Sherman Woodbury matrix identity.

    Parameters
    ----------
    X : torch.Tensor
        Data points (rows are observations, columns are variables).

    mu : torch.Tensor
        Mean vector of the multivariate normal distribution.

    Lambda : torch.Tensor
        Diagonal component of covariance matrix

    W : torch.Tensor
        Weight matrix.

    I_l : torch.Tensor, optional
        LxL identity matrix. If not provided, it is set to the
        identity matrix.

    Returns
    -------
    torch.Tensor
        Log probability of the data points under the multivariate
        normal distribution.
    """
    L, p = W.shape

    if I_l is None:
        I_l = torch.eye(L)

    term1 = -p * 0.9189385332046727

    X_demeaned = X - mu

    Wt = torch.transpose(W, 0, 1)

    F = la.solve(Lambda, Wt)
    E2 = I_l + W @ F

    ldet_L = torch.logdet(Lambda)
    ldet_E2 = torch.logdet(E2)
    term2 = -0.5 * (ldet_L + ldet_E2)

    end_center = la.solve(E2, F.T)
    middle = la.inv(Lambda) - F @ end_center
    end = middle @ X_demeaned.T

    quad = -1 * X_demeaned.T * end
    term3 = torch.sum(quad, axis=0)  # type: ignore

    log_prob_window = term1 + term2 + term3 / 2
    log_prob = torch.mean(log_prob_window)

    return log_prob
