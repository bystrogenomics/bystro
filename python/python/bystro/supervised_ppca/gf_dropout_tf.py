"""
This implements Factor analysis but with variational inference to allow us
to supervise a single latent variable to be predictive of an auxiliary 
outcome. Currently only implements logistic regression but in the future 
will be modified to allow for more general predictive outcomes.

Objects
-------
PPCADropouttf(PPCAtf)
    This is PPCA with SGD but supervises a single factor to predict y.

SPPCADropouttf(SPCAtf)
    This is SPCA with SGD but supervises a single factor. Currently 
    unimplemented since I need to get this PR in.

FactorAnalysisDropouttf(FactorAnalysistf)
    This is factor analysis but supervises a single factor. Currently 
    unimplemented since I need to get this PR in.

Methods
-------
_get_projection_matrix(W_,sigma_,n_components)
    Computes the parameters for p(S|X)

mvn_sample_tf(mu,Sigma,lamb=0.0)
    This samples from a normal distribution with potential tikhonov
    regularization to ensure positive definiteness
"""
import numpy as np
from tqdm import trange
from ._misc_tf import limit_gpu, return_optimizer_tf
from ._losses_tf import loss_cross_entropy_tf
from ._lprobs_tf import llike_mvnorm_tf

import tensorflow as tf
from tensorflow.keras.losses import (
    SparseCategoricalCrossentropy,
    MeanSquaredError,
)
from .gf_generative_tf import PPCAtf
from ._losses_tf import loss_offdiag_lq_tf


def _get_projection_matrix(W_, sigma_):
    """
    This is currently just implemented for PPCA due to nicer formula. Will
    modify for broader application later.

    Computes the parameters for p(S|X)

    Parameters
    ----------
    W_ : tf.Tensor(n_components,p)
        The loadings

    sigma_ : tf.Flaot
        Isotropic noise

    Returns
    -------
    Proj_X : tf.tensor(n_components,p)
        Beta such that Proj_XX = E[S|X]

    Cov : tf.tensor(n_components,n_components)
        Var(S|X)
    """
    n_components = int(W_.shape[0])
    eye = tf.constant(np.eye(n_components).astype(np.float32))
    M = tf.matmul(W_, tf.transpose(W_)) + sigma_ * eye
    Proj_X = tf.linalg.solve(M, W_)
    Cov = tf.linalg.inv(M) * sigma_
    return Proj_X, Cov


def mvn_sample_tf(mu, Sigma, lamb=0.0):
    """
    This samples from a normal distribution with potential tikhonov
    regularization to ensure positive definiteness

    Parameters
    ----------
    mu : tf.array,(?,p)
        The mean

    Sigma : tf.array,(?,p,p)
        The covariance matrix

    lamb : float,default=0.0
        The tikhonov regularization

    Returns
    -------
    z_sample : tf.Array,(?,p)
        The sampled values
    """
    if lamb == 0.0:
        chol = tf.linalg.cholesky(Sigma)
    else:
        p = Sigma.shape[-1]
        eye = tf.constant(np.eye(p).astype(np.float32))
        chol = tf.linalg.cholesky(Sigma + eye)

    eps = tf.random.normal(shape=mu.shape)
    if len(chol.shape) == 2:
        z_sample = tf.matmul(eps, tf.transpose(chol)) + mu
    else:
        eps_e = tf.expand_dims(eps, axis=-1)
        mul = tf.multiply(eps_e, tf.transpose(chol, perm=(0, 2, 1)))
        prod = tf.reduce_sum(mul, axis=1)
        z_sample = prod + mu

    return z_sample


class PPCADropouttf(PPCAtf):
    def __init__(
        self,
        n_components=2,
        n_supervised=1,
        prior_options={},
        mu=1.0,
        gamma=10.0,
        delta=5.0,
        training_options={},
    ):
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.n_supervised = int(n_supervised)
        super(PPCADropouttf, self).__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )

    def fit(self, X, y=None, task="classification"):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : np.array-like,(n_samples,n_prediction)
            Covariates we wish to predict. For now lazy and assuming
            logistic regression.

        task : string,default='classification'
            Is this prediction, multinomial regression, or classification

        Returns
        -------
        self : object
            The model
        """
        rng = np.random.default_rng(2021)
        td = self.training_options
        limit_gpu(td["gpu_memory"])
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        N, p = X.shape
        self.p = p

        if task == "classification":
            supervision_loss = loss_cross_entropy_tf
        elif task == "regression":
            supervision_loss = MeanSquaredError()
        else:
            supervision_loss = SparseCategoricalCrossentropy()

        W_, sigmal_ = self._initialize_variables(X)
        B_ = tf.Variable(0.0)
        trainable_variables = [W_, sigmal_, B_]

        self._initialize_saved_losses()
        self.losses_supervision = np.zeros(td["n_iterations"])

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])
        eye = tf.constant(np.eye(p).astype(np.float32))

        _prior = self._create_prior()

        for i in trange(td["n_iterations"]):
            idx = rng.choice(X.shape[0], size=td["batch_size"], replace=False)
            X_batch = X[idx]
            y_batch = y[idx]

            with tf.GradientTape() as tape:
                sigma = tf.nn.softplus(sigmal_)
                WWT = tf.matmul(tf.transpose(W_), W_)
                Sigma = WWT + sigma * eye

                like_prior = _prior(trainable_variables)

                like_gen = tf.reduce_mean(llike_mvnorm_tf(X_batch, Sigma))

                P_x, Cov = _get_projection_matrix(W_, sigma)
                mean_z = tf.matmul(X_batch, tf.transpose(P_x))
                z_samples = mvn_sample_tf(mean_z, Cov)
                y_hat = self.delta * z_samples[:, self.n_supervised] + B_

                loss_y = supervision_loss(y_batch, y_hat)
                WTW = tf.matmul(W_, tf.transpose(W_))

                loss_i = loss_offdiag_lq_tf(WTW)

                posterior = like_gen + 1 / N * like_prior
                loss = -1 * posterior + self.mu * loss_y + self.gamma * loss_i

            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            self._save_losses(i, like_gen, like_prior, posterior)
            self.losses_supervision[i] = loss_y.numpy()

        self._save_variables(trainable_variables)
        return self

    def _save_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of tensorflow variables saved

        Sets
        ----
        W_ : np.array-like,(n_components,p)
            The loadings

        sigmas_ : np.array-like,(n_components,p)
            The diagonal variances
        """
        self.W_ = trainable_variables[0].numpy()
        self.sigma2_ = tf.nn.softplus(trainable_variables[1]).numpy()
        self.B_ = trainable_variables[2].numpy()
