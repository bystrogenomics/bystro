"""
This implements canonical correlation analysis as described in Murphey 2010.
It is basically PPCA with the covariates divided into two groups. There are
shared latent components describing all the groupings.  

Objects
-------
CCA(BaseGDModel)

Methods
-------
None
"""
import numpy as np
import sklearn.decomposition as dp
from tqdm import trange
from ..utils._misc_tf import limit_gpu, return_optimizer_tf

import tensorflow as tf
from tensorflow.linalg import LinearOperatorFullMatrix
from tensorflow.linalg import LinearOperatorBlockDiag
from ..utils._lprobs_tf import llike_mvnorm_tf

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


class CCA(BaseGDModel):
    def __init__(
        self,
        n_components=2,
        n_components_i=1,
        prior_options={},
        training_options={},
    ):
        """
        The init method for CCA.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        n_components_i : int,default=2
            The latent dimensionality per group
        """
        super(CCA, self).__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self.n_components_i = int(n_components_i)

    def __repr__(self):
        out_str = "CCA object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X, groups=None):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            Data from population 1

        Returns
        -------
        self : object
            The model
        """
        td = self.training_options
        limit_gpu(td["gpu_memory"])
        rng = np.random.default_rng(2021)

        if groups is None:
            raise ValueError("Groups required for CCA")

        self.n_groups = len(np.unique(groups))
        self.groups = groups
        self.qs = np.zeros(self.n_groups)
        for i in range(self.n_groups):
            self.qs[i] = np.sum(groups == i)
        X = X.astype(np.float32)
        N, self.p = X.shape
        self.p = int(self.p)

        self.n_comps = self.n_components_i * np.ones(self.n_groups)

        trainable_variables = self._initialize_variables(X)
        W_ = trainable_variables[0]
        sigmal_ = trainable_variables[-1]
        Blist = trainable_variables[1:-1]

        eye = tf.constant(np.eye(self.p).astype(np.float32))

        _prior = self._create_prior()

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])

        self._initialize_saved_losses()

        for i in trange(td["n_iterations"]):
            idx = rng.choice(X.shape[0], size=td["batch_size"], replace=False)
            X_batch = X[idx]
            with tf.GradientTape() as tape:
                Bsquares = [tf.matmul(B, tf.transpose(B)) for B in Blist]
                B_block = block_diagonal_square_tf(Bsquares)
                WWT = tf.matmul(W_, tf.transpose(W_))
                sigma2 = tf.nn.softplus(sigmal_)
                Sigma = B_block + WWT + sigma2 * eye

                like_prior = _prior(trainable_variables)

                like_tot = tf.reduce_mean(llike_mvnorm_tf(X_batch, Sigma))
                posterior = like_tot + 1 / N * like_prior

                loss = -1 * posterior

            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            self._save_losses(i, like_tot, like_prior, posterior)

        self._save_variables(trainable_variables)

        return self

    def get_covariance(self):
        """
        Gets the covariance matrix (see Murhpy)

        Parameters
        ----------
        None

        Returns
        -------
        covariance : np.array-like(p,p)
            The covariance matrix
        """
        W = self._construct_w_tot()
        cov = self.sigma2_ * np.eye(self.p) + np.dot(W, W.T)
        return cov

    def _construct_w_tot(self):
        """
        This constructs the multigroup extension of the matrix in Murphy
        (12.93).

        Returns
        -------
        W_tot : np.array_like(self.p,sum(subspace)+shared)
            Total W with conveniently placed zeros
        """
        p = self.p
        dimX = int(np.sum(self.n_comps) + self.n_components)
        W_tot = np.zeros((p, dimX))
        if self.n_components == 1:
            W_tot[:, 0] = np.squeeze(self.W_)
        else:
            W_tot[:, : self.n_components] = self.W_.T
        count = self.n_components_i
        count2 = 0
        for i in range(self.n_groups):
            stx = int(count2)
            edx = int(count2 + self.qs[i])
            sty = int(count)
            edy = int(count + self.n_comps[i])
            W_tot[stx:edx, sty:edy] = self.B_list_[i]
            count += self.n_comps[i]
            count2 += self.qs[i]

        return W_tot

    def _initialize_variables(self, V):
        """
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        Returns
        -------
        var_list : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model
        """
        rng = np.random.default_rng(200)
        ncomp = self.n_comps
        mod = dp.PCA(self.n_components)
        mod.fit(V)
        W_ = tf.Variable(mod.components_.T.astype(np.float32))
        Bs = []
        for i in range(self.n_groups):
            bi = rng.normal(size=(int(self.qs[i]), int(ncomp[i]))).astype(
                np.float32
            )
            Bs.append(tf.Variable(bi))

        sigmal_ = tf.Variable(-3.0)
        var_list = [W_] + Bs + [sigmal_]
        return var_list

    def _create_prior(self):
        """
        For now we don't put a prior on our parameters until we can think
        of a reasonable set of parameters.

        """

        def _prior(trainable_variables):
            return 0.0 * tf.reduce_mean(trainable_variables[0])

        return _prior

    def _fill_prior_options(self, prior_options):
        return prior_options

    def _save_variables(self, tv):
        """ """
        self.W_ = tv[0].numpy()
        self.W_ = self.W_.T
        b_vals = tv[1:-1]
        self.B_list_ = [b_vals[i].numpy() for i in range(len(b_vals))]
        sigma2 = tf.nn.softplus(tv[-1])
        self.sigma2_ = sigma2.numpy()

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
