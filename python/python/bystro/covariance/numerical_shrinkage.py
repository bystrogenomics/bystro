import numpy as np
from scipy.linalg import sqrtm, logm

def numerical_shrinkage(eigenvals, gamma, loss):
    """
    Calculate optimal shrinker numerically.

    Parameters
    ----------
    eigenvals : array-like of shape (n,)
        A list of lambda values where the optimal shrinker should be calculated.
    gamma : float
        Aspect ratio.
    loss : str
        The desired loss function. Available loss functions:
            - 'F_1', 'F_2', 'F_3', 'F_4', 'F_5', 'F_6', 'F_7',
            - 'O_1', 'O_2', 'O_3', 'O_4', 'O_5', 'O_6', 'O_7', 
            - 'N_1', 'N_2', 'N_3', 'N_4', 'N_5', 'N_6', 'N_7', 
            - 'Stein', 'Ent', 'Div', 'Fre', 'Aff'

    Returns
    -------
    bestLam : array of shape (n,)
        The value of the optimal shrinker at the points given by eigenvals.
    bestLoss : array of shape (n,)
        The corresponding optimal loss.
    defaultLoss : array of shape (n,)
        The corresponding loss of hard thresholding (no shrinkage). 
    """
    bestLam = np.zeros_like(eigenvals)
    bestLoss = np.zeros_like(eigenvals)
    defaultLoss = np.zeros_like(eigenvals)
    for i, lam in enumerate(eigenvals):
        bestLam[i], bestLoss[i], defaultLoss[i] = numerical_shrinkage_impl(lam, gamma, loss)
    return bestLam, bestLoss, defaultLoss

def half(A, B):
    '''Computes the matrix square root of A and returns the product of the inverse square root of A with matrix B, followed by the inverse square root of A.'''
    return np.linalg.inv(np.sqrt(A)) @ B @ np.linalg.inv(np.sqrt(A))

Delta_1 = lambda A, B: A - B
Delta_2 = lambda A, B: np.linalg.inv(A) - np.linalg.inv(B)
Delta_3 = lambda A, B: np.linalg.inv(A) @ B - np.eye(A.shape[0])
Delta_4 = lambda A, B: np.linalg.inv(B) @ A - np.eye(A.shape[0])
Delta_6 = lambda A, B: half(A, B) - np.eye(A.shape[0])
Delta_7 = lambda A, B: logm(half(A, B))

def impl_F_1(A, B):
    '''Computes the Frobenius norm of the difference between matrices A and B.'''
    return np.linalg.norm(Delta_1(A, B), 'fro')

def impl_F_2(A, B):
    '''Computes the Frobenius norm of the difference between the inverses of matrices A and B.'''
    return np.linalg.norm(Delta_2(A, B), 'fro')

def impl_F_3(A, B):
    '''Computes the Frobenius norm of the difference between the product of the inverse of matrix A with matrix B and the identity matrix.'''
    return np.linalg.norm(Delta_3(A, B), 'fro')

def impl_F_4(A, B):
    '''Computes the Frobenius norm of the difference between the product of the inverse of matrix B with matrix A and the identity matrix.'''
    return np.linalg.norm(Delta_4(A, B), 'fro')

def impl_F_6(A, B):
    '''Computes the Frobenius norm of the difference between half of the matrix product of the inverse square root of matrix A with matrix B and the identity matrix.'''
    return np.linalg.norm(Delta_6(A, B), 'fro')

def impl_O_1(A, B):
    '''Computes the operator norm of the difference between matrices A and B.'''
    return np.linalg.norm(Delta_1(A, B))

def impl_O_2(A, B):
    '''Computes the operator norm of the difference between the inverses of matrices A and B.'''
    return np.linalg.norm(Delta_2(A, B))

def impl_O_6(A, B):
    '''Computes the operator norm of the difference between half of the matrix product of the inverse square root of matrix A with matrix B and the identity matrix.'''
    return np.linalg.norm(Delta_6(A, B))

def impl_N_1(A, B):
    '''Computes the nuclear norm of the difference between matrices A and B.'''
    return np.linalg.norm(np.linalg.svd(Delta_1(A, B)), 1)

def impl_N_2(A, B):
    '''Computes the nuclear norm of the difference between the inverses of matrices A and B.'''
    return np.linalg.norm(np.linalg.svd(Delta_2(A, B)), 1)

def impl_N_3(A, B):
    '''Computes the nuclear norm of the difference between the product of the inverse of matrix A with matrix B and the identity matrix.'''
    return np.linalg.norm(np.linalg.svd(Delta_3(A, B)), 1)

def impl_N_4(A, B):
    '''Computes the nuclear norm of the difference between the product of the inverse of matrix B with matrix A and the identity matrix.'''
    return np.linalg.norm(np.linalg.svd(Delta_4(A, B)), 1)

def impl_N_6(A, B):
    '''Computes the nuclear norm of the difference between half of the matrix product of the inverse square root of matrix A with matrix B and the identity matrix.'''
    return np.linalg.norm(np.linalg.svd(Delta_6(A, B)), 1)

def impl_Stein(A, B):
    Delta = np.linalg.inv(np.sqrt(A)) @ B @ np.linalg.inv(np.sqrt(A))
    return (np.trace(Delta) - np.log(np.linalg.det(Delta)) - np.trace(np.eye(A.shape[0]))) / 2

def impl_Ent(A, B):
    '''Computes the Entropy loss between matrices A and B.'''
    return impl_Stein(B, A)

def impl_Div(A, B):
    '''Computes the Divergence loss between matrices A and B.'''
    return np.trace(np.linalg.inv(A) @ B + np.linalg.inv(B) @ A - 2 * np.eye(A.shape[0])) / 2.0

def impl_Fre(A, B):
    '''Computes the Fidelity loss between matrices A and B.'''
    return np.trace(A + B - 2.0 * (sqrtm(A) @ sqrtm(B)))

def impl_Aff(A, B):
    '''Computes the Affine loss between matrices A and B.'''
    return 0.5 * np.log(0.5 * np.linalg.det(A + B) / (np.sqrt(np.linalg.det(A)) * np.sqrt(np.linalg.det(B))))

def numerical_shrinkage_impl(lam, gamma, loss):
    """
    Compute the optimal regularization parameter, corresponding loss value, and default loss value using numerical shrinkage.

    Parameters
    ----------
    lam : float
        Regularization parameter.
    gamma : float
        Shrinkage parameter.
    loss : str
        The type of loss function. It must be one of {'Stein', 'Ent', 'Div', 'Fre', 'Aff'}.

    Returns
    -------
    optLam : float
        Optimal regularization parameter.
    optVal : float
        Corresponding optimal loss value.
    defaultRisk : float
        Default loss value.
    """
    lam_plus = (1 + np.sqrt(gamma))**2

    if lam < lam_plus:
        raise ValueError('lambda below bulk edge')

    ell = lambda lam: np.where(lam >= lam_plus, ((lam + 1 - gamma) + np.sqrt((lam + 1 - gamma)**2 - 4 * lam)) / 2.0, 0)

    c = lambda lam: np.where(lam >= lam_plus, np.sqrt((1 - gamma / ((ell(lam) - 1)**2)) / (1 + gamma / (ell(lam) - 1))), 0)



    A = np.array([[ell(lam), 0], [0, 1]])

    LossFunc = globals()[f'impl_{loss}']

    optLam, optVal = bestLam_impl(A, c(lam), LossFunc)
    defaultRisk = defaultLoss_impl(lam, A, c(lam), LossFunc)

    return optLam, optVal, defaultRisk

def bestLam_impl(A, c, J):
    """
    Compute the optimal regularization parameter and corresponding loss value using a nested grid search approach.

    Parameters:
    -----------
    A : numpy.ndarray
        The input matrix A.
    c : float
        Coefficient value.
    J : function
        Loss function to be optimized.

    Returns:
    --------
    optLam : float
        Optimal regularization parameter.
    optVal : float
        Corresponding optimal loss value.
    """
    lobnd = 1
    upbnd = A[0, 0] + 2
    s = -np.sqrt(1 - c**2)
    u = np.array([c, s])
    optVal = float('inf')
    optLam = 0

    for _ in range(6):
        lamList = np.linspace(lobnd, upbnd, 100)
        inx = 0
        for iLam in range(len(lamList)):
            lam = lamList[iLam]
            eta = lam - 1
            B = np.eye(A.shape[0]) + eta * np.outer(u, u)
            val = J(A, B)
            if val < optVal:
                optVal = val
                optLam = lam
                inx = iLam
        if inx > 0:
            loinx = max(0, inx - 1)
            hiinx = min(inx + 1, len(lamList) - 1)
            lobnd = lamList[loinx]
            upbnd = lamList[hiinx]
        else:
            break

    return optLam, optVal


def defaultLoss_impl(lam, A, c, J):
    """
    Compute the default loss value for a given regularization parameter.

    Parameters:
    -----------
    lam : float
        Regularization parameter.
    A : numpy.ndarray
        The input matrix A.
    c : float
        Coefficient value.
    J : function
        Loss function.

    Returns:
    --------
    risk : float
        Default loss value.
    """
    s = -np.sqrt(1 - c**2)
    u = np.array([c, s])
    A1 = np.eye(A.shape[0]) + (lam - 1) * np.outer(u, u)
    risk = J(A, A1)
    return risk

def SteinLoss(A, B):
    """
    Compute the Stein loss for matrices A and B.

    Parameters:
    -----------
    A : numpy.ndarray
        Input matrix A.
    B : numpy.ndarray
        Input matrix B.

    Returns:
    --------
    J : float
        Stein loss value.
    """
    Delta = np.linalg.inv(sqrtm(A)) @ B @ np.linalg.inv(sqrtm(A))
    J = (np.trace(Delta) - np.log(np.linalg.det(Delta)) - np.trace(np.eye(A.shape[0]))) / 2
    return J