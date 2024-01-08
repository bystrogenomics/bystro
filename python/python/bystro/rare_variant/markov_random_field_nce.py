"""
This implements a Markov Random Field for identifying conditional 
relationships in rare variants. The model is similar to an ising model 
and performs inference using noise contrastive estimation. The model we 
have posited is 

p(x_1,...,x_p) prop exp(Phi_jx_j +  Theta_jk x_jx_k)
Theta_jk >= 0

where x_i is a bernoulli random variable indicating the presence of a 
variant.

Background
----------

Noise contrastive Estimation:
Noise-contrastive estimation (NCE) is a technique used in machine learning,
particularly in the context of training models for probabilistic language 
modeling or other generative tasks. It was introduced as an alternative to 
traditional maximum likelihood estimation (MLE) for training models with a 
large number of parameters. NCE is often employed when working with models 
like neural networks in scenarios where computing the partition function 
(also known as normalization constant) is computationally expensive or 
intractable.

Here's a high-level explanation of how noise-contrastive estimation works:

Objective Function:
In traditional maximum likelihood estimation, the goal is to maximize the 
likelihood of the observed data given the model parameters. This involves 
maximizing the probability assigned by the model to the true data.
In contrast, NCE transforms the problem into a binary classification 
task. Instead of directly modeling the probability of the data, NCE 
introduces a binary classification objective.

Noise Distribution:
NCE introduces a noise distribution, which is typically chosen to be a 
simple distribution (e.g., uniform or Gaussian). This distribution 
represents "noise" samples that are not part of the true data distribution.

Positive and Negative Samples:
For each training example (positive sample) from the true data distribution,
NCE generates a set of negative samples from the noise distribution. These 
negative samples serve as contrastive examples during training.

Binary Classification Task:
The objective becomes a binary classification problem where the model is 
trained to distinguish between positive samples (real data) and negative 
samples (noise). The model is adjusted to assign high probabilities to 
true data and low probabilities to noise.

Loss Function:
The loss function in NCE is derived from the binary classification task. 
It is typically based on logistic regression and aims to minimize the 
negative log-likelihood of the true data being positive and the noise 
being negative.

Parameter Updates:
The model parameters are updated using techniques such as stochastic 
gradient descent (SGD) to minimize the loss function over the training 
data. The goal is to adjust the model parameters so that it becomes good 
at discriminating between true data and noise.
By framing the problem as a binary classification task and using noise 
samples as contrastive examples, NCE provides a computationally efficient 
way to train generative models, especially in scenarios where calculating 
the partition function is challenging. 

Objects
-------
MarkovRandomFieldNCE(BaseSGDModel)
    This implements a Markov Random Field model of rare variants as an 
    undirected graphical model


Methods
-------
None

"""
import numpy as np
from numpy.typing import NDArray

from tqdm import trange
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from bystro.rare_variant._base_mrf import BaseMarkovRandomField


class MarkovRandomFieldNCE(BaseMarkovRandomField):
    """
    This implements a Markov Random Field model for rare variant 
    characterization using Noise Contrastive Estimation as an 
    inference method
    """

    def fit(
        self,
        X: NDArray[np.float_],
        progress_bar: bool = True,
        seed: int = 2021,
    ):
        """
        Fits a model given covariates X 

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        Returns
        -------
        self : MarkovRandomField
            The model
        """
        self._test_inputs(X)
        training_options = self.training_options
        prior_options = self.prior_options
        n_noise = int(training_options["nu"] * training_options["batch_size"])
        N, p = X.shape
        self.p = p

        rng = np.random.default_rng(int(seed))
        marginal_probs = np.mean(X, axis=0)

        Phi_, L_l, log_z = self._initialize_variables(X)
        trainable_variables = [Phi_, L_l, log_z]

        X_tensor = self._transform_training_data(X.astype(np.float32))[0]

        optimizer = torch.optim.Adam(
            trainable_variables, lr=training_options["learning_rate"]
        )

        relu = nn.ReLU()

        m_phi = MultivariateNormal(
            prior_options["mu_phi"] * torch.ones(p),
            prior_options["sigma_phi"] * torch.eye(p),
        )
        m_L = nn.MSELoss()
        zeros = torch.zeros(p, p)

        mp = torch.tensor(marginal_probs.astype(np.float32))

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            idx = rng.choice(
                X_tensor.shape[0],
                size=training_options["batch_size"],
                replace=False,
            )
            X_batch = X_tensor[idx]
            Y_gen = rng.binomial(1, marginal_probs, size=(n_noise, p))
            Y_batch = torch.tensor(Y_gen.astype(np.float32))

            L = torch.tril(relu(L_l), diagonal=-1)
            Theta = 0.5 * (L + torch.transpose(L, 0, 1))

            log_prior = m_phi.log_prob(Phi_) + m_L(L_l, zeros)

            XT = torch.matmul(X_batch, Theta)
            quad_x = torch.sum(XT * X_batch, dim=1)
            vec_x = torch.matmul(X_batch, Phi_)
            nll_x = quad_x + vec_x
            log_p_m_x = -1 * nll_x + log_prior
            log_p_n_x_point = mp * X_batch + (1 - mp) * (1 - X_batch)
            log_p_n_x = torch.sum(log_p_n_x_point, dim=1)
            G_x = log_p_m_x - log_p_n_x
            h_x = 1 / (1 + training_options["nu"] * torch.exp(-G_x))

            YT = torch.matmul(Y_batch, Theta)
            quad_y = torch.sum(YT * Y_batch, dim=1)
            vec_y = torch.matmul(Y_batch, Phi_)
            nll_y = quad_y + vec_y
            log_p_m_y = -1 * nll_y + log_prior
            log_p_n_y_point = mp * Y_batch + (1 - mp) * (1 - Y_batch)
            log_p_n_y = torch.sum(log_p_n_y_point, dim=1)
            G_y = log_p_m_y - log_p_n_y
            h_y = 1 / (1 + training_options["nu"] * torch.exp(-G_y))

            loss = torch.sum(torch.log(h_x)) + torch.sum(torch.log(1 - h_y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, loss)

        self._store_instance_variables(trainable_variables)

        return self
