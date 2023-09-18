"""
This maximizes the marginal likelihood for the Poisson model, for now only 
a Possion is implemented. The generative model is 

p(z) = Dir(alpha_k)
p(x_ij|z=k) = Poisson(Lambda_{jk})

The log likelihood is 

log p(x) = log(\\sum p(z=k)p(x_ij|z=k)). 

There's not an analytic formula that makes this easy to solve. But as long
as you use the logsumexp trick you can compute this easily numerically. I 
honestly don't know why that isn't taught more, I only learned it as a 
postdoc even though I got a PhD in statistics. Meh


Objects
-------
MVTadaPoissonML(K=4,training_options={})
    This fits a standard Poisson latent variable model.

Methods
-------
None

"""
import numpy as np
import torch
from tqdm import trange  # type: ignore
from torch import nn
from torch.distributions import Dirichlet
import scipy.stats as st  # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from torch.nn import PoissonNLLLoss

from bystro._template_sgd_np import BaseSGDModel  # type: ignore


class MVTadaPoissonML(BaseSGDModel):
    def __init__(self, K=4, training_options=None):
        """
        This is a product Poisson latent variable model.

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

        self.pi = None
        self.Lambda_ = None

    def fit(self, data, progress_bar=True, Lamb_init=None, pi_init=None):
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
        self._test_inputs(data)
        training_options = self.training_options
        rng = np.random.default_rng(2021)

        N, p = data.shape

        lambda_latent, pi_logits = self._initialize_variables(
            data, Lamb_init, pi_init
        )
        X = self._transform_training_data(data)[0]
        trainable_variables = [lambda_latent, pi_logits]

        m_d = Dirichlet(torch.tensor(np.ones(self.K) * self.K))

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        nll = PoissonNLLLoss(full=True, log_input=False, reduction="none")
        mse = nn.MSELoss()
        smax = nn.Softmax()
        softplus = nn.Softplus()

        myrange = trange if progress_bar else range

        for i in myrange(training_options["n_iterations"]):
            idx = rng.choice(N, training_options["batch_size"], replace=False)
            X_batch = X[idx]  # Batch size x

            Lambda_ = softplus(lambda_latent)
            mu = 0.01
            pi_ = smax(pi_logits) * mu + (1 - mu) / self.K

            loss_logits = 0.001 * mse(pi_logits, torch.zeros(self.K))
            loss_prior_pi = -1.0 * m_d.log_prob(pi_) / N

            loglikelihood_each = [
                -1 * nll(X_batch, Lambda_[k]) for k in range(self.K)
            ]  # List of K N x p log likelihoods

            loglikelihood_sum = [
                torch.sum(mat, axis=1) for mat in loglikelihood_each
            ]  # List of K N x 1 log likelihoods

            loglikelihood_stack = torch.stack(loglikelihood_sum)
            # Matrix of N x k log likelihoods
            loglikelihood_components = torch.transpose(
                loglikelihood_stack, 0, 1
            ) + torch.log(
                pi_
            )  # Matrix of Nxk posteriors
            loglikelihood_marg = torch.logsumexp(
                loglikelihood_components, dim=1
            )  # Vector of N x 1 marginal posteriors
            loss_likelihood = -1 * torch.mean(loglikelihood_marg)
            # Average marginal likelihood

            loss = loss_logits + loss_prior_pi + loss_likelihood
            self._save_losses(i, loss_likelihood, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._store_instance_variables(trainable_variables)
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
        default_options = {
            "n_iterations": 3000,
            "batch_size": 200,
            "learning_rate": 5e-5,
            "momentum": 0.99,
        }
        tops = {**default_options, **training_options}
        return tops

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
        rng = np.random.default_rng(2021)

        # Initialize variables
        if Lamb_init is None:
            model = GaussianMixture(self.K)
            model.fit(X)
            Lamb_init = model.means_.astype(np.float32)
        if pi_init is None:
            pi_init = rng.dirichlet(np.ones(self.K)).astype(np.float32)
        lambda_latent = torch.tensor(Lamb_init, requires_grad=True)
        pi_logits = torch.tensor(pi_init, requires_grad=True)

        return lambda_latent, pi_logits

    def _initialize_save_losses(self):
        """
        This method initializes the arrays to track relevant variables
        during training

        Parameters
        ----------
        """
        self.losses_likelihood = np.empty(self.training_options["n_iterations"])
        self.losses_total = np.empty(self.training_options["n_iterations"])

    def _save_losses(self, i, negative_log_likelihood, loss):
        """
        This saves the respective losses at each iteration

        Parameters
        ----------
        """
        self.losses_likelihood[i] = negative_log_likelihood.detach().numpy()
        self.losses_total[i] = loss.detach().numpy()

    def _store_instance_variables(self, trainable_variables):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of variables to save 
        """
        smax = nn.Softmax()
        softplus = nn.Softplus()
        mu = 0.01
        pi_ = smax(trainable_variables[1]) * mu + (1 - mu) / self.K
        Lambda_ = softplus(trainable_variables[0])
        self.Lambda = Lambda_.detach().numpy()
        self.pi = pi_.detach().numpy()

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
