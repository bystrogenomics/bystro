"""
This implements Gaussian factor analysis models with generative (standard)
inference. There are three different models. Gaussian factor analysis 
parameterizes the covariance matrix as 

    Sigma = WW^T + Lambda

where Lambda is a diagonal matrix. The parameters of the diagonal determine
the model. Probabilistic principal component analysis sets Lambda =
sigma**2*I_p. Supervised principal component analysis parameterizes it as 
Lambda = diag([sigma_x**2*1_p,sigma_y**21_q]), allowing for variances to 
differ between the predictive and dependent variables. Finally, factor 
analysis allows each diagonal component to be distinct. Models (1) and (3)
are described in Bishop 2006, while supervised Probabilistic PCA is 
described in several papers, including Yu 2006.

Objects
-------
PPCAtf(BaseGDModel)
    Principal component analysis but with tensorflow implementation.

SPCAtf(BaseGDModel)
    Supervised probabilistic component analysis

FactorAnalysistf(BaseGDModel)
    Factor analysis implemented in tensorflow. See Bishop 2006

Methods
-------
None
"""
import numpy as np
from tqdm import trange
from ._misc_tf import limit_gpu, return_optimizer_tf
from ._lprobs_tf import llike_gamma_tf, llike_mvnorm_tf
from ._misc_np import softplus_inverse_np
from sklearn import decomposition as dp
import deepcopy

import tensorflow as tf
from ._base import BaseGDModel


class PPCAtf(BaseGDModel):
    def __init__(self, n_components=2, prior_options={}, training_options={}):
        """
        This implements probabilistic PCA with stochastic gradient descent.
        There are two benefits over the standard baseline method (1) it
        allows for priors to be placed on the parameters. (2) is minor but
        it theoretically allows for larger datsets that can't be loaded into
        memory. More importantly, it makes a fairer baseline comparison
        for my other models because it allows for stochastic noise.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        training_options : dict,default={}
            The options for gradient descent

        prior_options : dict,default={}
            The options for priors on model parameters
        """
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )

    def __repr__(self):
        out_str = "PPCAtf object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : None
            Used for model consistency

        Returns
        -------
        self : object
            The model
        """
        rng = np.random.default_rng(2021)
        td = self.training_options
        limit_gpu(td["gpu_memory"])
        X = X.astype(np.float32)
        N, p = X.shape
        self.p = p

        W_, sigmal_ = self._initialize_variables(X)
        trainable_variables = [W_, sigmal_]

        self._initialize_saved_losses()

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])
        eye = tf.constant(np.eye(p).astype(np.float32))

        _prior = self._create_prior()

        for i in trange(td["n_iterations"]):
            idx = rng.choice(X.shape[0], size=td["batch_size"], replace=False)
            X_batch = X[idx]

            with tf.GradientTape() as tape:
                sigma = tf.nn.softplus(sigmal_)
                WWT = tf.matmul(tf.transpose(W_), W_)
                Sigma = WWT + sigma * eye

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
        covariance = np.dot(self.W_.T, self.W_) + self.sigma2_ * np.eye(self.p)
        return covariance

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the log density of the prior
        """
        pd = self.prior_options
        if pd["type"] == "gamma":

            def log_prior(tv):
                return -1 * llike_gamma_tf(tv[1], pd["alpha"], pd["beta"])

        elif pd["type"] == "None":

            def log_prior(tv):
                return 0 * llike_gamma_tf(tv[1], pd["alpha"], pd["beta"])

        else:
            raise NotImplementedError("Option %s unimplemented" % pd["type"])
        return log_prior

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        if "type" not in prior_options:
            prior_options["type"] = "gamma"
        if prior_options["type"] == "gamma":
            defaults = {"alpha": 1.0, "beta": 1.0}
        elif prior_options["type"] == "None":
            defaults = {}
        else:
            raise NotImplementedError(
                "Prior %s not implemented" % prior_options["type"]
            )

        new_dict = deepcopy(defaults)
        new_dict.update(prior_options)
        return new_dict

    def _initialize_variables(self, X):
        """
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        Returns
        -------
        W_ : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : tf.Float
            The variance of the model
        """
        model = dp.PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = tf.Variable(W_init)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = tf.Variable(sinv)
        return W_, sigmal_

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

        sigma2_ : float
            The isotropic variance
        """
        self.W_ = trainable_variables[0].numpy()
        self.sigma2_ = tf.nn.softplus(trainable_variables[1]).numpy()


class SPCAtf(BaseGDModel):
    def __init__(self, n_components=2, prior_options={}, training_options={}):
        """
        This implements supervised probabilistic component analysis. Unlike
        PPCA there are no analytic solutions for this model. While the
        initial paper used expectation maximization as an inference method,
        EM is actually pretty bad so this is a way better way to go.

        SPPCA replaces isotropic noise with noise for groups of variables.
        The paper Yu et al (2006) only has two groups of covariates, but
        my implementation is more general in that it allows for multiple
        groups rather than two.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        training_options : dict,default={}
            The options for gradient descent

        prior_options : dict,default={}
            The options for priors on model parameters
        """
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )

    def __repr__(self):
        out_str = "SPCAtf object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X, groups=None):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : None
            Used for model consistency

        groups : np.array-like,(n_covariates,)
            Divide the covariates into groups with different isotropic noise

        Returns
        -------
        self : object
            The model
        """
        rng = np.random.default_rng(2021)
        td = self.training_options
        limit_gpu(td["gpu_memory"])
        X = X.astype(np.float32)
        N, p = X.shape
        self.p = p

        self.n_groups = len(np.unique(groups))

        W_, slist = self._initialize_variables(X)
        trainable_variables = [W_] + slist

        clist = self._get_const_list(groups)

        self._initialize_saved_losses()

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])

        _prior = self._create_prior()

        for i in trange(td["n_iterations"]):
            idx = rng.choice(X.shape[0], size=td["batch_size"], replace=False)
            X_batch = X[idx]

            with tf.GradientTape() as tape:
                WWT = tf.matmul(tf.transpose(W_), W_)
                D = tf.linalg.diag(
                    tf.add_n(
                        [
                            tf.nn.softplus(slist[i]) * clist[i]
                            for i in range(self.n_groups)
                        ]
                    )
                )
                Sigma = WWT + D

                like_prior = _prior(trainable_variables)
                like_tot = tf.reduce_mean(llike_mvnorm_tf(X_batch, Sigma))
                posterior = like_tot + 1 / N * like_prior
                loss = -1 * posterior

            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            self._save_losses(i, like_tot, like_prior, posterior)

        self._save_variables(W_, D)
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
        covariance = np.dot(self.W_.T, self.W_) + np.diag(self.sigmas_)
        return covariance

    def _get_const_list(self, idxs):
        """
        This is an auxiliary method that generates tf constants to
        efficiently construct the diagonal isotropic noise matrix.

        Parameters
        ----------
        idxs : np.array(self.p)
            The groupings of the covariates

        Returns
        -------
        myList : list[tf.constant-like(n_covariates)]
            A list of constant vector of ones used for construction.
        """
        myList = []
        for i in range(self.n_groups):
            numpy_array = np.zeros(self.p)
            numpy_array[idxs == i] = 1
            myList.append(tf.constant(numpy_array.astype(np.float32)))
        return myList

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the negative log density of the prior
        """
        pd = self.prior_options
        if pd["type"] == "gamma":

            def log_prior(tv):
                loss = 0.0
                for i in range(len(tv) - 1):
                    loss += llike_gamma_tf(tv[1 + i], pd["alpha"], pd["beta"])
                return loss

        elif pd["type"] == "None":

            def log_prior(trainable_variables):
                return 0.0 * trainable_variables[0]

        else:
            raise NotImplementedError("Option %s unimplemented" % pd["type"])
        return log_prior

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        if "type" not in prior_options:
            prior_options["type"] = "None"
        if prior_options["type"] == "gamma":
            defaults = {"alpha": 1.0, "beta": 1.0}
        elif prior_options["type"] == "None":
            defaults = {}
        else:
            raise NotImplementedError(
                "Prior %s not implemented" % prior_options["type"]
            )
        new_dict = deepcopy(defaults)
        new_dict.update(prior_options)
        return new_dict

    def _initialize_variables(self, X):
        """
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        Returns
        -------
        W_ : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : list[tf.Variable]
            A list of the isotropic noises for each group
        """
        model = dp.PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = tf.Variable(W_init)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = [tf.Variable(sinv) for i in range(self.n_groups)]
        return W_, sigmal_

    def _save_variables(self, W_, D_):
        """
        Saves the learned variables

        Parameters
        ----------
        W_ : tf.Tensor,(n_components,p)
            The loadings

        D_ : tf.Tensor,(p,p)
            Diagonal matrix of isotropic noises

        Sets
        ----
        W_ : np.array-like,(n_components,p)
            The loadings

        sigmas_ : np.array-like,(n_components,p)
            The diagonal variances
        """
        self.W_ = W_.numpy()
        self.sigmas_ = tf.linalg.diag_part(D_).numpy()


class FactorAnalysistf(BaseGDModel):
    def __init__(self, n_components=2, prior_options={}, training_options={}):
        """
        This implements factor analysis which allows for each covariate to
        have it's own isotropic noise. No analytic solution that I know of
        but fortunately with SGD it doesn't matter.

        Parameters
        ----------
        n_components : int,default=2
            The latent dimensionality

        training_options : dict,default={}
            The options for gradient descent

        prior_options : dict,default={}
            The options for priors on model parameters
        """
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )

    def __repr__(self):
        out_str = "FactorAnalysistf object\n"
        out_str += "n_components=%d\n" % self.n_components
        return out_str

    def fit(self, X):
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : np.array-like,(n_samples,n_covariates)
            The data

        y : None
            Used for model consistency

        Returns
        -------
        self : object
            The model
        """
        rng = np.random.default_rng(2021)
        td = self.training_options
        limit_gpu(td["gpu_memory"])
        X = X.astype(np.float32)
        N, p = X.shape
        self.p = p

        W_, sigmal_ = self._initialize_variables(X)
        trainable_variables = [W_, sigmal_]

        self._initialize_saved_losses()

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])

        _prior = self._create_prior()

        for i in trange(td["n_iterations"]):
            idx = rng.choice(X.shape[0], size=td["batch_size"], replace=False)
            X_batch = X[idx]

            with tf.GradientTape() as tape:
                sigmas = tf.nn.softplus(sigmal_)
                WWT = tf.matmul(tf.transpose(W_), W_)
                D = tf.linalg.diag(sigmas)
                Sigma = WWT + D

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
        covariance = np.dot(self.W_.T, self.W_) + np.diag(self.sigmas_)
        return covariance

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the negative log density of the prior
        """
        pd = self.prior_options
        if pd["type"] == "gamma":

            def log_prior(tv):
                return tf.reduce_mean(
                    llike_gamma_tf(tv[1], pd["alpha"], pd["beta"])
                )

        elif pd["type"] == "None":

            def log_prior(tv):
                return 0.0 * tf.reduce_mean(
                    llike_gamma_tf(tv[1], pd["alpha"], pd["beta"])
                )

        else:
            raise NotImplementedError("Option %s unimplemented" % pd["type"])
        return log_prior

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        if "type" not in prior_options:
            prior_options["type"] = "gamma"
        if prior_options["type"] == "gamma":
            defaults = {"alpha": 1.0, "beta": 1.0}
        elif prior_options["type"] == "None":
            defaults = {}
        else:
            raise NotImplementedError(
                "Prior %s not implemented" % prior_options["type"]
            )
        new_dict = deepcopy(defaults)
        new_dict.update(prior_options)
        return new_dict

    def _initialize_variables(self, X):
        """
        Initializes the variables of the model

        Parameters
        ----------
        X : np.array-like,(n_samples,p)
            The data

        Returns
        -------
        W_ : tf.Variable-like,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : tf.Float
            The variance of the model
        """
        model = dp.PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = tf.Variable(W_init)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = tf.Variable(sinv[0] * np.ones(self.p).astype(np.float32))
        return W_, sigmal_

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
        self.sigmas_ = tf.nn.softplus(trainable_variables[1]).numpy()


"""
MIT License

Copyright (c) 2022 Austin Talbot

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
