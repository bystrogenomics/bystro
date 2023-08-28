"""
This maximizes the marginal likelihood for the Poisson model, for now only 
a Possion is implemented. The generative model is 

p(z) = Dir(alpha_k)
p(x_ij|z=k) = Poisson(Lambda_{jk})

The log likelihood is 

log p(x) = log(\sum p(z=k)p(x_ij|z=k)). 

There's not an analytic formula that makes this easy to solve. But as long
as you use the logsumexp trick you can compute this easily numerically. I 
honestly don't know why that isn't taught more, I only learned it as a 
postdoc even though I got a PhD in statistics. Meh


Objects
-------
MVTadaPoissonML(K=4,training_options={})
    This fits a standard poisson latent variable model.

Methods
-------
None

"""
import numpy as np
import torch
from tqdm import trange
from torch import nn
from torch.distributions import Dirichlet
from copy import deepcopy
import scipy.stats as st
from sklearn.mixture import GaussianMixture
from torch.nn import PoissonNLLLoss


class MVTadaPoissonML(object):
    def __init__(self, K=4, training_options={}):
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
        self.training_options = self._fill_training_options(training_options)

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
        td = self.training_options
        K = self.K

        N, p = data.shape
        X = torch.tensor(data)
        rng = np.random.default_rng(2021)

        # Initialize variables
        if Lamb_init is None:
            model = GaussianMixture(K)
            model.fit(data)
            Lamb_init = model.means_.astype(np.float32)
        if pi_init is None:
            pi_init = rng.randn(K).astype(np.float32)
        lambda_latent = torch.tensor(Lamb_init, requires_grad=True)
        pi_logits = torch.tensor(pi_init, requires_grad=True)

        trainable_variables = [lambda_latent, pi_logits]

        m_d = Dirichlet(torch.tensor(np.ones(self.K) * self.K))

        optimizer = torch.optim.SGD(
            trainable_variables, lr=td["learning_rate"], momentum=0.99
        )

        nll = PoissonNLLLoss(full=True, log_input=False, reduction="none")
        mse = nn.MSELoss()
        smax = nn.Softmax()
        softplus = nn.Softplus()

        myrange = trange if progress_bar else range

        self.losses_likelihoods = np.zeros(td["n_iterations"])

        for i in myrange(td["n_iterations"]):
            idx = rng.choice(N, td["batch_size"], replace=False)
            X_batch = X[idx]  # Batch size x

            Lambda_ = softplus(lambda_latent)
            mu = 0.01
            pi_ = smax(pi_logits) * mu + (1 - mu) / K

            loss_logits = 0.001 * mse(
                pi_logits, torch.zeros(K)
            )  # Don't allow explosions
            loss_prior_pi = -1.0 * m_d.log_prob(pi_) / N

            loglikelihood_each = [
                -1 * nll(X_batch, Lambda_[k]) for k in range(K)
            ]

            loglikelihood_sum = [
                torch.sum(mat, axis=1) for mat in loglikelihood_each
            ]  

            loglikelihood_stack = torch.stack(loglikelihood_sum)
            loglikelihood_components = torch.transpose(
                loglikelihood_stack, 0, 1
            ) + torch.log(pi_)
            loglikelihood_marg = torch.logsumexp(
                loglikelihood_components, dim=1
            )
            loss_likelihood = -1 * torch.mean(loglikelihood_marg)

            loss = loss_logits + loss_prior_pi + loss_likelihood
            self.losses_likelihoods[i] = loss_likelihood.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.Lambda = Lambda_.detach().numpy()
        self.pi = pi_.detach().numpy()

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
            "n_iterations": 30000,
            "batch_size": 200,
            "learning_rate": 5e-5,
        }
        tops = deepcopy(default_options)
        tops.update(training_options)
        return tops
