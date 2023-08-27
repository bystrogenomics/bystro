"""
This implements partial least squares according to Murphy 2010. This allows
for a small number of factors to be covariate-exclusive, explaining 
irrelevant variance while allowing the shared components to predict the 
response variables.

Objects
-------
PLS(n_components=2,n_components_x=1,prior_options={},training_options={})

Methods
-------
None
"""
import numpy as np
from ._misc_tf import return_optimizer_tf

import tensorflow as tf
from tensorflow.linalg import LinearOperatorFullMatrix
from tensorflow.linalg import LinearOperatorBlockDiag

from ._base import BaseGDModel


def block_diagonal_square_tf(matrices):
    """
    Constructs a block diagonal matrix from a list of square matrices

    Parameters
    ----------
    matrices : list
        List of square matrices in tensorflow

    Returns
    -------
    block_matrix : tf.Tensor,shape=(p,p)
        The matrices in block diagonal form
    """
    linop_blocks = [LinearOperatorFullMatrix(block) for block in matrices]
    linop_block_diagonal = LinearOperatorBlockDiag(linop_blocks)
    block_matrix = linop_block_diagonal.to_dense()
    return block_matrix


class PLS(BaseGDModel):
    def __init__(
        self,
        n_components=2,
        n_components_x=1,
        prior_options={},
        training_options={},
    ):
        """
        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        n_components_x : int,default=1
            The number of covariates specific to the predictors.

        Sets
        ----
        creationDate : datetime
            The date/time that the object was created
        """
        super(PLS, self).__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self.n_components_x = int(n_components_x)

    def __repr__(self):
        out_str = "PLS object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X, y=None):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            Data from population 1

        y : np.array-like,(n_samples,p)
            Data from population 2

        Returns
        -------
        self : object
            The model
        """
        rng = np.random.default_rng(2021)
        td = self.training_options
        V = np.hstack((X, y))
        V = V.astype(np.float32)
        N, p = X.shape
        q = y.shape[1]
        self.p, self.q = p, q

        W_, B_x_, sigmal_ = self._initialize_variables()
        trainable_variables = [W_, B_x_, sigmal_]

        zeros_Y = tf.constant(np.zeros((q, q)).astype(np.float32))
        eye = tf.constant(np.eye(p + q).astype(np.float32))

        self._initialize_saved_losses()

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])

        _prior = self._create_prior()

        for i in range(td["n_iterations"]):
            idx = rng.choice(V.shape[0], size=td["batch_size"], replace=False)
            V_batch = V[idx]
            with tf.GradientTape() as tape:
                Bsquare = tf.matmul(B_x_, tf.transpose(B_x_))
                B_block = block_diagonal_square_tf([Bsquare, zeros_Y])
                WWT = tf.matmul(W_, tf.transpose(W_))
                sigma2 = tf.nn.softplus(sigmal_)
                Sigma = B_block + WWT + sigma2 * eye

                like_prior = _prior(trainable_variables)

                like_tot = tf.reduce_mean(llike_mvnorm_tf(V_batch, Sigma))
                posterior = like_tot + 1 / N * like_prior

                loss = -1 * posterior
            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            self._save_losses(i, like_tot, like_prior, posterior)

        self._save_variables(trainable_variables)

        return self

    def get_covariance(self):
        """
        Gets the covariance matrix

        Sigma = W^TW + sigma2*I

        Parameters
        ----------
        None

        Returns
        -------
        covariance : np.array-like(p,p)
            The covariance matrix
        """
        W = self._construct_w_tot()
        cov = self.sigma2_ * np.eye(self.p + self.q) + np.dot(W, W.T)
        return cov

    def _construct_w_tot(self):
        """ """
        dimX = np.sum(self.n_components + self.n_components_x)
        W_tot = np.zeros((self.p + self.q, dimX))
        if self.n_components == 1:
            W_tot[:, 0] = np.squeeze(self.W_)
        else:
            W_tot[:, : self.n_components] = self.W_

        W_tot[: self.p, self.n_components :] = self.B_x_
        return W_tot

    def _initialize_variables(self):
        """
        Initializes the variables of the model

        Returns
        -------
        W_x : tf.Variable, array-like (n_features_x,n_components)
            The shared loadings for x

        W_y : tf.Variable, array-like (n_features_y,n_components)
            The shared loadings for y

        B_x : tf.Variable, array-like (n_features_x,n_components_)
            The exclusive loadings for x
        """
        rng = np.random.default_rng(2021)
        Ls, Lx = self.n_components, self.n_components_x
        p, q = self.p, self.q
        W_ = tf.Variable(rng.normal(size=(p + q, Ls)).astype(np.float32))
        B_x = tf.Variable(rng.normal(size=(p, Lx)).astype(np.float32))
        sigmal_ = tf.Variable(-2.0)
        return W_, B_x, sigmal_

    def _create_prior(self):
        """
        For now we don't put a prior on our parameters until we can think
        of a reasonable set of parameters.

        """

        def _prior(trainable_variables):
            return 0.0 * tf.reduce_mean(trainable_variables[0])

        return _prior

    def _save_variables(self, tv):
        """ """
        self.W_ = tv[0].numpy()
        self.B_x_ = tv[1].numpy()
        sigma2 = tf.nn.softplus(tv[2])
        self.sigma2_ = sigma2.numpy()

    def _fill_prior_options(self):
        """
        Fills in parameters used for prior of parameters

        Paramters
        ---------
        prior_dict: dictionary
            The prior parameters used to specify the prior

        Options
        -------
        sig_BX: float, L2 penalization on Bx
        sig_WX: float, L2 penalization on Wx
        sig_BY: float, L2 penalization on By
        sig_WY: float, L2 penalization on Wy

        """
        return {"sig_BX": 0.01, "sig_WX": 0.01, "sig_WY": 0.01}

    def transform(self, X):
        """
        This returns the latent variable estimates given X

        Parameters
        ----------
        X : np array-like,(N_samples,\sum idxs)
            The data to transform

        Returns
        -------
        S : np.array-like,(N_samples,n_components)
            The factor estimates
        """
        raise NotImplementedError("Transform not implemented yet")


def llike_mvnorm_tf(X, Sigma, safe=False):
    """
    This returns log likelihood of

    p(x|Sigma)=|2pi Sigma|^{-1/2}exp(-1/2(x)^TSigma^{-1}(x))

    Parameters
    ----------
    x : array,shape=(n_data,p)
        The data

    Sigma : array,shape=(p,p)
        The covariance matrix

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    llike : tf.array,shape=(n_data,)
        The log likelihood
    """
    p = Sigma.shape[1]
    if safe:
        Sigma = tf.cast(Sigma, tf.float32)
        X = X.astype(np.float32)
    term1 = -p / 2 * tf.math.log(2 * np.pi)
    term2 = -0.5 * tf.linalg.logdet(Sigma)
    quad_init = tf.linalg.solve(Sigma, tf.transpose(X))
    diff = tf.multiply(X, tf.transpose(quad_init))
    row_sum = tf.reduce_sum(diff, axis=1)
    term3 = -1 / 2 * row_sum
    llike = term1 + term2 + term3
    return llike
