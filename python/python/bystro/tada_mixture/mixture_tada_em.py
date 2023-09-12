"""
This uses the expectation maximization algorithm to fit a latent variable
model for TADA, with two flavors, a Zero-Inflated Poisson (ZIP) and a 
standard Poisson. The generative formulation is 

p(z) = Dir(alpha_k)
p(x_ij|z=k) = (ZI)Poisson(Lambda_{jk})

Obviously the ZIP will have a parameter pi for probability of 0.

ZIP does not have an analytic maximum likelihood formula. This is not a 
problem for EM,  the M step only requires that we be able to maximize
the likelihood, not that it has an analytic solution. So we just use 
standard gradient descent with automatic differentiation in Torch. This 
could be naughty if the objective was multimodal, but I strongly doubt that
this is the case.

Objects
-------
MVTadaPoissonEM(K=4,training_options={})
    This fits a standard Poisson latent variable model.

MVTadaZipEM(K=4,training_options={})
    This fits a standard ZIP latent variable model.

Methods
-------
None

"""
import numpy as np
import scipy.stats as st  # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from tqdm import trange  # type: ignore

import torch
from torch import nn
from torch.nn import PoissonNLLLoss
from bystro._template_sgd_np import BaseSGDModel  # type: ignore


class MVTadaPoissonEM(BaseSGDModel):
    def __init__(self, K=4, training_options=None):
        """
        This is a Product Poisson latent variable model.

        Parameters
        ----------
        K : int,default=4
            The number of clusters, i.e. number of different types of
            genes

        training_options : dict,default={}
            The parameters for the inference scheme
        """
        self.K = int(K)
        super().__init__(training_options=training_options)
        self._initialize_save_losses()

        self.pi_ = None
        self.Lambda_ = None

    def fit(self, X, progress_bar=True, Lamb_init=None, pi_init=None):
        """
        Fits a model given count data X.

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The count data

        progress_bar : bool,default=True
            Whether to print a progress bar while fitting

        Lamb_init : np.array-like,shape=(k,p),default=None
            Initialize the loadings of the model. Defaults to fitting a
            GMM on the data and using those.

        pi_init : np.array-like,shape=(k),default=None
            Initialize the class weights of the model.

        Returns
        -------
        self
        """
        self._test_inputs(X)
        training_options = self.training_options

        N, p = X.shape

        pi_, Lambda_ = self._initialize_variables(Lamb_init, pi_init)

        myrange = trange if progress_bar else range

        for i in myrange(training_options["n_iterations"]):
            # E step compute most likely categories
            z_probs = np.zeros((N, self.K))
            for j in range(self.K):
                log_pmf = st.poisson.logpmf(X, Lambda_[j])
                log_pmf_sum = np.sum(log_pmf, axis=1)
                z_probs[:, j] = log_pmf_sum + np.log(pi_[j])
            expectation_z = np.argmax(z_probs, axis=1)

            for j in range(self.K):
                pi_[j] = np.mean(expectation_z == j)
                Lambda_[j] = np.mean(X[expectation_z == j], axis=0)

            eps = 0.001
            pi_ = pi_ * (1 - self.K * eps) + np.ones(self.K) * eps

            self._save_losses(
                i,
            )

        self._store_instance_variables([pi_, Lambda_])
        return self

    def _store_instance_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of variables to save
        """
        self.pi_ = trainable_variables[0]
        self.Lambda_ = trainable_variables[1]

    def _initialize_variables(self, X, Lamb_init, pi_init):
        """
        This initializes the factor loadings (Lambda), and the component
        probabilities (pi). If Lamb_init  or pi_init are given these are
        the values obviously. Otherwise, the initialization of the weights
        is random and the Loadings are a Gaussian mixture model.

        Parameters
        ----------

        Returns
        -------
        """
        # Initialize variables
        if Lamb_init is None:
            model = GaussianMixture(self.K)
            model.fit(X)
            Lambda_ = model.means_
        else:
            Lambda_ = Lamb_init

        rng = np.random.default_rng(2021)
        pi_ = rng.dirichlet(np.ones(self.K)) if pi_init is None else pi_init

        return Lambda_, pi_

    def _initialize_save_losses(self):
        """
        This method initializes the arrays to track relevant variables
        during training

        TODO

        Parameters
        ----------
        """

    def _save_losses(self):
        """
        This saves the respective losses at each iteration

        TODO

        Parameters
        ----------
        """

    def _test_inputs(self, X):
        """
        This performs error checking on inputs for fit

        Parameters
        ----
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data must be numpy array")

    def _transform_training_data(self, *args):
        """
        Not needed as this object does not use pytorch for inference

        Parameters
        ----------
        """

    def predict(self, data):
        """
        This predicts the latent cluster assignments given a fitted model

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        log_proba = self.predict_logproba(data)
        z_hat = np.argmax(log_proba, axis=1)
        return z_hat

    def predict_logproba(self, data):
        """
        Predicts the log probability a datapoint being assigned a specific
        cluster.

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        N = data.shape[0]
        log_proba = np.zeros((N, self.K))
        for i in range(N):
            for k in range(self.K):
                log_proba[i, k] = np.sum(
                    st.poisson.logpmf(data[i], self.Lambda[k])
                )
        return log_proba

    def _fill_training_options(self, training_options):
        """
        This fills any relevant parameters for the learning algorithm

        Parameters
        ----------
        training_options : dict

        Returns
        -------
        tops : dict
        """
        default_options = {"n_iterations": 100}
        tops = {**default_options, **training_options}
        return tops


class MVTadaZipEM(BaseSGDModel):
    def __init__(self, K=4, training_options=None):
        """
        This is a Product ZIP latent variable model.

        Parameters
        ----------
        K : int,default=4
            The number of clusters, i.e. number of different types of
            genes

        training_options : dict,default={}
            The parameters for the inference scheme
        """
        self.K = K
        super().__init__(training_options=training_options)
        self._initialize_save_losses()

        self.pi_ = None
        self.Lambda = None
        self.Alpha = None

    def fit(self, data, progress_bar=True):
        """
        Fits a model given count data X.

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The count data
        Returns
        -------
        self
        """
        self._test_inputs(data)
        N, p = data.shape
        training_options = self.training_options
        K = self.K

        Lambda_list_l, alpha_ls, pi_ = self._initialize_variables(data)

        sigmoid = nn.Sigmoid()

        X = self._transform_training_data(data)[0]

        myloss = PoissonNLLLoss(log_input=False, full=True, reduction="none")
        myrange = trange if progress_bar else range

        for i in myrange(training_options["n_iterations"]):
            # E step compute most likely categories
            z_probs = np.zeros((N, K))
            for k in range(K):
                Lambda_k = Lambda_list_l[k]
                alpha_k = sigmoid(alpha_ls[k])  # Prob of being 0

                log_likelihood_poisson = -1 * myloss(Lambda_k, X)
                log_likelihood_poisson_w = log_likelihood_poisson + torch.log(
                    1 - alpha_k
                )

                log_likelihood_0 = torch.log(
                    alpha_k + (1 - alpha_k) * torch.exp(-1 * Lambda_k)
                )
                log_likelihood_0_b = torch.broadcast_to(
                    log_likelihood_0, data.shape
                )

                log_likelihood_point = torch.where(
                    data != 0, log_likelihood_poisson_w, log_likelihood_0_b
                )

                log_marginal = torch.sum(log_likelihood_point, axis=1)

                z_probs_k = log_marginal + np.log(pi_[k])
                z_probs[:, k] = z_probs_k.detach().numpy()

            Ez = np.argmax(z_probs, axis=1)

            # M step, maximize parameters
            for k in range(K):
                pi_[k] = np.mean(Ez == k)
                data_sub = data[Ez == k]  # Selects relevant data

                trainable_variables = [Lambda_list_l[k], alpha_ls[k]]

                optimizer = torch.optim.SGD(
                    trainable_variables,
                    lr=training_options["learning_rate"],
                    momentum=0.8,
                )

                for j in range(training_options["n_inner_iterations"]):
                    Lambda_k = Lambda_list_l[k]
                    alpha_k = sigmoid(alpha_ls[k])

                    log_likelihood_poisson = -1 * myloss(Lambda_k, data_sub)
                    log_likelihood_poisson_w = (
                        log_likelihood_poisson + torch.log(1 - alpha_k)
                    )

                    log_likelihood_0 = torch.log(
                        alpha_k + (1 - alpha_k) * torch.exp(-1 * Lambda_k)
                    )
                    log_likelihood_0_b = torch.broadcast_to(
                        log_likelihood_0, data_sub.shape
                    )

                    log_likelihood_point = torch.where(
                        data_sub != 0,
                        log_likelihood_poisson_w,
                        log_likelihood_0_b,
                    )

                    log_marginal = torch.sum(log_likelihood_point, axis=1)
                    loss = -1 * torch.mean(log_marginal)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    Lambda_k.requires_grad_(False)
                    Lambda_k[Lambda_k < 0] = 0
                    Lambda_k.requires_grad_(True)

        self._store_instance_variables([pi_, Lambda_list_l, alpha_ls])

        return self

    def predict(self, data):
        """
        This predicts the latent cluster assignments given a fitted model

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        log_proba = self.predict_logproba(data)
        z_hat = np.argmax(log_proba, axis=1)
        return z_hat

    def predict_logproba(self, data):
        """
        Predicts the log probability a datapoint being assigned a specific
        cluster.

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        N = data.shape[0]
        log_proba = np.zeros((N, self.K))
        for i in range(N):
            for k in range(self.K):
                log_prob_poisson = st.poisson.logpmf(data[i], self.Lambda[k])
                log_prob_0 = np.log(
                    self.Alpha[k]
                    + (1 - self.Alpha[k]) * np.exp(-1 * self.Lambda[k])
                )
                probs = log_prob_poisson.copy()
                probs[data[i] == 0] = log_prob_0[data[i] == 0]
                log_proba[i, k] = np.sum(probs)
        return log_proba

    def _store_instance_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of variables to save
        """
        self.pi = trainable_variables[0]
        Lambda_list_l = trainable_variables[1]
        alpha_ls = trainable_variables[2]
        sigmoid = nn.Sigmoid()
        for k in range(self.K):
            Lambda_k = Lambda_list_l[k]
            alpha_k = sigmoid(alpha_ls[k])

            self.Lambda[k] = Lambda_k.detach().numpy()
            self.Alpha[k] = alpha_k.detach().numpy()

    def _initialize_variables(self, X):
        """

        Parameters
        ----------

        Returns
        -------
        """
        model = GaussianMixture(self.K)
        model.fit(X)
        Lambda_list_l = []
        for i in range(self.K):
            Lambda_list_l.append(
                torch.tensor(
                    1.2 * model.means_[i].astype(np.float32) + 0.1,
                    requires_grad=True,
                )
            )

        pct_0 = np.mean(X == 0, axis=0)
        pct_0 = np.clip(pct_0, 0.01, 0.99)
        pct_0_inv = np.log(pct_0 / (1 - pct_0))

        alpha_ls = [
            torch.tensor(pct_0_inv, requires_grad=True) for i in range(self.K)
        ]
        pi_ = 1 / self.K * np.ones(self.K)
        return Lambda_list_l, alpha_ls, pi_

    def _initialize_save_losses(self):
        """
        This method initializes the arrays to track relevant variables
        during training

        TODO

        Parameters
        ----------
        """

    def _save_losses(self):
        """
        This saves the respective losses at each iteration

        TODO

        Parameters
        ----------
        """

    def _test_inputs(self, X):
        """
        This performs error checking on inputs for fit

        Parameters
        ----
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data must be numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")

    def _transform_training_data(self, *args):
        """
        Not needed as this object does not use pytorch for inference

        Parameters
        ----------
        """
        out = []
        for arg in args:
            out.append(torch.tensor(arg))
        return out

    def _fill_training_options(self, training_options):
        """
        This fills any relevant parameters for the learning algorithm

        Parameters
        ----------
        training_options : dict

        Returns
        -------
        tops : dict
        """
        default_options = {
            "n_iterations": 2000,
            "n_inner_iterations": 50,
            "learning_rate": 5e-4,
        }
        tops = {**default_options, **training_options}
        return tops
