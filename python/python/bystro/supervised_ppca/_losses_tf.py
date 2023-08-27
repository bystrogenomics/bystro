"""
This implements some common losses and methods used to create new losses 

Methods
-------

##################
# Generic losses #
##################
loss_cosine_similarity_tf(vec1,vec2)
    Computes the absolute value of cosine similarity (encourages orthogonality)

loss_cross_entropy_tf(X,X_hat,weights=None)
    Computes the cross entropy between y_true and y_est

def loss_offdiag_lq_tf(Sigma,q=2):
    Computes the Lq loss on the off-diagonal elements

######################
# Huber-based losses #
######################

loss_huber_p_tf(x,xhat=None,p=1,delta=1.0,weights=None)
    Loss(x,x_hat) = { 1/2(x-x_hat)**2, abs(x-x_hat)\le delta <= delta
                    { 1/p*delta**(2-p)*abs(x-x_hat)**p + (1/2-1/p)*delta**2
                                                    otherwise

#################
# Normed losses #
#################
loss_norm_nuclear_tf(X)
    This computes the nuclear norm via an SVD. This is the convex relaxation
    of rank(X).

loss_norm_lq_tf(X,Xhat=None,q=2,weights=None)
    Computes the Lq loss 

############################
# Negative log likelihoods #
############################
nll_beta_tf(x,alpha,beta,weights=None)
    p(x|a,b) = x^{alpha-1}(1-x)^{beta-1)Ga(alpha+beta)/(Ga(alpha)Ga(beta))

nll_binom_tf(x,N,p,weights=None)
    p(x|N,p) = (n choose x) p^x(1-p)^{n-x}

nll_exp_tf(x,lamb,weights=None)
    p(x|lamb) = lamb*exp(-lamb*x)

nll_gamma_tf(x,a,b,weights=None)
    p(x|alpha,beta) = beta^alpha/Gamma(alpha)x^{alpha-1}e^{-beta x}

nll_gaussian_tf(x,mu,sigma,weights=None)
    p(x,mu,Sigma) = 1/sqrt(2pi sigma**2)exp((x-mu)**2/2sigma)

nll_igamma_tf(x,a,b,weights=None)
    p(x|alpha,beta) = beta^alpha/Gamma(alpha)x^{-alpha-1}exp(-beta/x)

nll_mvnorm_tf(X,mu,Sigma,weights=None)
    p(x|mu,Sigma)=|2pi Sigma|^{-1/2}exp(-1/2(x-mu)^TSigma^{-1}(x-mu))

nll_negative_binom_tf(x,p,r,weights=None)
    p(x|r,p) = (x + r - 1 choose x) (1-p)^r p^x

nll_poisson_tf(x,lamb,weights=None)
    p(x|lamb)=lamb^x exp(-lamb)/x!

######################
# Truncation methods #
######################
truncate_loss_flat(loss,k)
    This truncates any loss to be below a threshold k 

truncate_loss_sigmoid(loss,k)
    This uses the sigmoid function to truncate a loss below a threshold 
    that maintains differentiability.
"""
import tensorflow as tf
import numpy as np
from ._lprobs_tf import (
    llike_beta_tf,
    llike_binom_tf,
    llike_exp_tf,
    llike_gamma_tf,
    llike_gaussian_tf,
    llike_igamma_tf,
    llike_mvnorm_tf,
    llike_negative_binom_tf,
    llike_poisson_tf,
)

######################
######################
##                  ##
##  Generic losses  ##
##                  ##
######################
######################


def loss_cosine_similarity_tf(vec1, vec2, eps=1e-4, safe=True):
    """
    Computes the absolute value of cosine similarity (encourages orthogonality)

    csim_abs = abs(vec1 dot vec2/norm(vec1)*norm(vec2))

    Paramters
    ---------
    vec1 : array,size=(k, or 1)
        First vector

    vec2 : array,size=(k, or 1)
        First vector

    eps : float,default=1e-4
        A fudge factor to prevent division by 0

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    csim_abs : tf.Float
        Absolute value of cross-entropy
    """
    if safe:
        vec1 = tf.cast(vec1, tf.float32)
        vec2 = tf.cast(vec2, tf.float32)
    v1 = tf.squeeze(vec1)
    v2 = tf.squeeze(vec2)
    num = tf.reduce_mean(tf.multiply(v1, v2))
    denom = tf.linalg.norm(v1) * tf.linalg.norm(v2)
    csim_abs = tf.abs(num / (denom + eps))
    return csim_abs


def loss_cross_entropy_tf(y_true, y_est, weights=None, safe=True):
    """
    Computes the cross entropy between y_true and y_est

    Paramters
    ---------
    y_true : array,size=(k,)
        True labels [1,0]

    y_est : array,size=(k,)
        Logits

    weights : array,size=(k,)
        The relative weights

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    loss : tf.Float
        The cross entropy loss
    """
    if safe:
        y_est = tf.cast(y_est, tf.float32)
    yt = tf.squeeze(y_true)
    ye = tf.squeeze(y_est)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=ye)
    loss = (
        tf.reduce_mean(ce) if weights is None else tf.reduce_mean(weights * ce)
    )
    return loss


def loss_huber_p_tf(x, xhat=None, p=1, delta=1.0, weights=None, safe=True):
    """
    Computes the loss

    Loss(x,x_hat) = { 1/2(x-x_hat)**2, abs(x-x_hat)\le delta <= delta
                    { 1/p*delta**(2-p)*abs(x-x_hat)**p + (1/2-1/p)*delta**2
                                                    otherwise

    The intuitive explanation is the loss is quadratic in the range [-delta,
    delta] and grows with exponent p outside that range, and is continuous.

    Parameters
    ----------
    x : array-like,(N_samples,1)
        The data

    x : array-like,(N_samples,1)
        Optional baseline data, defaults to 0

    p : float>0
        The exponent of the error outside of [-delta,delta]

    delta : float>0
        The range where the loss is quadratic

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    loss : tf.Float
        The loss
    """
    if safe:
        x = x.astype(np.float32)

    if xhat is None:
        xhat = 0.0

    error = x - xhat
    abs_error = tf.abs(x - xhat)
    if p == 1:
        losses = tf.where(
            abs_error <= delta,
            0.5 * tf.square(error),
            delta * abs_error - 0.5 * tf.square(delta),
        )
    elif p == 2:
        losses = 0.5 * tf.square(error)
    else:
        k = 1 / p * delta ** (2 - p)
        c = (1 / 2 - 1 / p) * delta ** 2
        losses = tf.where(
            abs_error <= delta, 0.5 * tf.square(error), k * abs_error ** p + c
        )
    loss = (
        tf.reduce_mean(losses)
        if weights is None
        else tf.reduce_mean(weights * losses)
    )
    return loss


#############
#############
##         ##
##  Norms  ##
##         ##
#############
#############


def loss_norm_nuclear_tf(X, safe=True):
    """
    This computes the nuclear norm via an SVD. This is the convex relaxation
    of rank(X).

    Paramters
    ---------
    X : tf.matrix,shape=(m,n)
        The input matrix

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    norm : tf.Float
        The estimated norm
    """
    if safe:
        X = tf.cast(X, tf.float32)
    s = tf.linalg.svd(X, compute_uv=False)
    norm = tf.reduce_sum(s)
    return norm


def loss_norm_lq_tf(X, Xhat=None, q=2, weights=None, safe=True):
    """
    Computes the Lq loss (omitting 1/q)

    Parameters
    ----------
    X : tf.array-like(p,r)
        Matrix of data

    Xhat : tf.array-like(p,r)
        Optional matrix of predictions (default=0)

    q : float>0
        The power

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    loss : tf.Float
        The loss
    """
    if safe:
        X = X.astype(np.float32)

    if Xhat is None:
        Xhat = 0

    if len(X.shape) > 1:
        if q == 1:
            loss_un = tf.reduce_mean(tf.abs(X - Xhat), axis=1)
        elif q == 2:
            loss_un = tf.reduce_mean(tf.square(X - Xhat), axis=1)
        else:
            loss_un = tf.reduce_mean(tf.math.pow(tf.abs(X - Xhat), q), axis=1)
    else:
        if q == 1:
            loss_un = tf.abs(X - Xhat)
        elif q == 2:
            loss_un = tf.square(X - Xhat)
        else:
            loss_un = tf.math.pow(tf.abs(X - Xhat), q)

    loss = (
        tf.reduce_mean(loss_un)
        if weights is None
        else tf.reduce_mean(weights * loss_un)
    )

    return loss


def loss_offdiag_lq_tf(Sigma, q=2, safe=True):
    """
    Computes the Lq loss on the off-diagonal elements

    Parameters
    ----------
    Sigma : tf.array-like(p,p)
        Square matrix

    q : float>0
        The power

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    loss : tf.Float
        The loss
    """
    if safe:
        Sigma = tf.cast(Sigma, tf.float32)
    D = tf.linalg.diag(tf.linalg.diag_part(Sigma))
    S_D = Sigma - D
    if q == 1:
        loss = tf.reduce_mean(tf.abs(S_D))
    elif q == 2:
        loss = tf.reduce_mean(tf.square(S_D))
    else:
        loss = tf.reduce_mean(tf.math.pow(tf.abs(S_D), q))
    return loss


###############################
###############################
##                           ##
##  Negative log likelihoods ##
##                           ##
###############################
###############################


def nll_beta_tf(x, alpha, beta, weights=None):
    """
    This returns the (weighted) average negative log likelihood of

    p(x|a,b) = (x^(alpha-1)) * ((1-x)^(beta-1)) * 
                            Ga(alpha+beta) / (Ga(alpha) * Ga(beta))

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    alpha : array,shape(1 or n_data,)
        The firstparameter

    beta : array,shape(1 or n_data,)
        The second parameter

    weights : array,default=None
        The weights to place on each observation

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_beta_tf(x, alpha, beta)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_binom_tf(x, N, p, weights=None):
    """
    This returns the (weighted) average negative log likelihood of

    p(x|N,p) = (n choose x) p^x(1-p)^{n-x}

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    N : array,shape(1 or n_data,)
        The number of trials

    p : array,shape(1 or n_data,)
        The probability of success

    weights : array,default=None
        The weights to place on each observation

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_binom_tf(x, N, p)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_exp_tf(x, lamb, weights=None, safe=True):
    """
    This returns the (weighted) average negative log likelihood of

    p(x|lamb) = lamb*exp(-lamb*x)

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    lamb : array,shape(1 or n_data,)
        The exponential parameter

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_exp_tf(x, lamb, safe=safe)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_gamma_tf(x, alpha, beta, weights=None, safe=True):
    """
    This returns the (weighted) average negative log likelihood of

    p(x|alpha,beta) = beta^alpha/Gamma(alpha)x^{alpha-1}e^{-beta x}

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    alpha : array,shape(1 or n_data,)
        The shape parameter(s)

    beta : array,shape(1 or n_data,)
        The rate parameter(s)

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_gamma_tf(x, alpha, beta, safe=safe)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_gaussian_tf(x, mu, sigma2, weights=None, safe=True):
    """
    This returns average negative log likelihood of

    p(x,mu,Sigma) = 1/sqrt(2pi sigma**2)exp((x-mu)**2/2sigma)

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    mu : array,shape(1 or n_data,)
        The mean parameter(s)

    sigma2 : array,shape(1 or n_data,)
        The variance parameter(s)

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_gaussian_tf(x, mu, sigma2, safe=safe)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_igamma_tf(x, alpha, beta, weights=None, safe=True):
    """
    This returns average negative log likelihood of

    p(x|alpha,beta) = beta^alpha/Gamma(alpha)x^{-alpha-1}exp(-beta/x)

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    alpha : array,shape(1 or n_data,)
        The shape parameter(s)

    beta : array,shape(1 or n_data,)
        The rate parameter(s)

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_igamma_tf(x, alpha, beta, safe=safe)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_mvnorm_tf(X, Sigma, weights=None, safe=True):
    """
    This returns average negative log likelihood of

    p(x|mu,Sigma)=|2pi Sigma|^{-1/2}exp(-1/2(x)^TSigma^{-1}(x))

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    weights : array,default=None
        The weights to place on each observation

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_mvnorm_tf(X, Sigma, safe=safe)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_negative_binom_tf(x, p, r, weights=None, safe=True):
    """
    This returns average negative log likelihood of

    p(x|r,p) = (x + r - 1 choose x) (1-p)^r p^x

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    p : array,shape=(n_data,) or (1,)
        Probability of success

    r : array,shape=(n_data,) or (1,)
        Number of failures

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    nll : tf.Float
        The (average)(weighted) negative log likelihood
    """
    lprobs = llike_negative_binom_tf(x, p, r, safe=safe)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


def nll_poisson_tf(x, lamb, weights=None):
    """
    This returns the (weighted) average negative log likelihood of

    p(x|lamb)=lamb^x exp(-lamb)/x!

    Parameters
    ----------
    x : array,shape=(n_data,)
        The data

    lamb : array,shape=(1,) or (n_data)
        The rate parameter

    weights : array,default=None
        The weights to place on each observation

    Returns
    -------
    nll : tf.Float
        The (average) negative log likelihood
    """
    lprobs = llike_poisson_tf(x, lamb)
    nll = _computed_weighted_nll(lprobs, weights)
    return nll


##########################
##########################
##                      ##
##  Truncation methods  ##
##                      ##
##########################
##########################


def truncate_loss_flat(loss, k):
    """
    This truncates any loss to be below a threshold k

    Parameters
    ----------
    loss : tf.array
        The previously-evaluated loss

    k : float
        The loss threshold

    Returns
    -------
    trunc : tf.array
        The truncated loss
    """
    trunc = tf.math.minimum(loss, k)
    return trunc


def truncate_loss_sigmoid(loss, k):
    """
    This uses the sigmoid function to truncate a loss below a threshold
    that maintains differentiability.

    Parameters
    ----------
    loss : tf.array
        The previously-evaluated loss

    k : float
        The loss threshold

    Returns
    -------
    trunc : tf.array
        The truncated loss
    """
    return 2 * (k * (tf.math.sigmoid(loss)) - k / 2)


def _computed_weighted_nll(lprobs, weights):
    """
    This takes the log likelihoods and weights them and negates to yield
    the average negative log likelihood

    Parameters
    ----------
    lprobs : tf.array,shape=(n_data,)

    weights : None or tf.array,shape=(n_data,)
        The optional weights

    Returns
    -------
    nll : tf.float
        The average negative log likelihood
    """
    nll = (
        -1 * tf.reduce_mean(lprobs)
        if weights is None
        else -1 * tf.reduce_mean(weights * lprobs)
    )
    return nll
