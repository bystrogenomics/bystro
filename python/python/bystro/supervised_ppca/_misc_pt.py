"""

Objects
-------
None

Methods
-------

ldet_sw_full

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
    ldet_A = torch.logdet(A)

    # Compute the log determinant of matrix B
    ldet_B = torch.logdet(B)

    # Compute the second term involving matrix V, A, and U
    term2 = torch.matmul(V, la.solve(A, U))

    # Compute the inverse of matrix B
    term1 = la.inv(B)

    # Compute the log determinant of the matrix product
    ldet_prod = la.logdet(term1 + term2)

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


def ldet_sw_factor_analysis(Lambda,W,I_l=None):
    ldet_L = la.logdet(Lambda)
    LiW = la.solve(Lambda,torch.transpose(W,0,1))
    WtLiW = torch.matmul(W,LiW)
    IWtLiW = I_l + WtLiW
    ldet_p = la.logdet(IWtLiW)
    log_determinant = ldet_p + ldet_l
    return log_determinant


def inv_sw_factor_analysis(Lambda,W,I_l=None,I_p=None):
    if I_l is None:
        I_l = torch.eye(W.shape[0]) 
    if I_p is None:
        I_p = torch.eye(W.shape[1])

    Lambda_inv = torch.inv(Lambda)
    WLi = torch.matmul(W,Lambda_inv)
    inner = I_l + torch.matmul(WLi,torch.transpose(W,0,1))
    inner_inv = la.inv(inner)
    end = torch.matmul(inner_inv,WLi)
    term2 = torch.matmul(torch.transpose(W,0,1),end)
    Imterm2 = I_p - term2
    Sigma_inv = torch.matmul(Lambda_inv,Imterm2)
    return Sigma_inv

def mvn_log_prob_sw(X,mu,Lambda,W,I_l=None):
    L,p = W.shape
    if I_l is None:
        I_l = torch.eye(L)

    term1 = -p*0.9189385332046727

    X_demeaned = X - mu

    B = la.solve(Lambda,X_demeaned)
    Wt = torch.transpose(W,0,1)
    C = la.solve(Lambda,W)
    D = torch.matmul(Wt,C)
    E = I_l + D


    ldet_L = la.logdet(Lambda)
    ldet_E = la.logdet(E)
    term2 = -0.5*(ldet_L + ldet_E)

    WtB = torch.matmul(Wt,B)
    end = la.solve(E,Wtb)
    CeiWtb = torch.matmul(C,end)
    quad_end = B - CeiWtb
    quad = -1*X*quad_end
    term3 = torch.sum(quad,axis=0)

    log_prob_window = term1 + term2 + term3
    log_prob = torch.mean(log_prob_window)
    return log_prob
    
