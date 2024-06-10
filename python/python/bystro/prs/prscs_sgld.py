import numpy as np

from tqdm import trange

import torch
from torch import nn

from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.linear_model import Ridge

from bystro.stochastic_gradient_langevin.sgld_optimizer_pt import (
    PreconditionedSGLDynamicsPT,
)
from bystro.stochastic_gradient_langevin.sgld_scheduler import (
    scheduler_sgld_geometric,
)

from bystro._template_sgld import BaseSGLDModel

ptd = torch.distributions
device = torch.device("cpu")
torch.set_default_tensor_type("torch.FloatTensor")


class PRSCS(BaseSGLDModel):
    def __init__(self, training_options=None, prior_options=None):
        self.sample_list = []
        super().__init__(
            training_options=training_options, prior_options=prior_options
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_bar: bool = True,
        seed: int = 2021,
    ) -> None:
        self._test_inputs(X, y)
        X_, y_ = self._transform_training_data(X, y)
        N, p = X.shape
        self.p = p
        training_options = self.training_options
        prior_options = self.prior_options

        rng = np.random.default_rng(int(seed))

        beta_, psi_l, delta_l, sigma2_l = self._initialize_variables(X, y)

        self.samples_beta = np.zeros((training_options["n_samples"], self.p))
        self.samples_psi = np.zeros((training_options["n_samples"], self.p))
        self.samples_delta = np.zeros((training_options["n_samples"], self.p))
        self.samples_sigma2 = np.zeros((training_options["n_samples"], self.p))

        var_list = [beta_, psi_l, delta_l, sigma2_l]

        f_prior_delta = ptd.Gamma(
            prior_options["b"], 1 / prior_options["phi"]
        ).log_prob

        lr_fn = scheduler_sgld_geometric(
            n_samples=training_options["n_samples"]
        )
        optimizer = PreconditionedSGLDynamicsPT(
            var_list, lr=0.001, weight_decay=0.5
        )
        zeros_p = torch.tensor(np.zeros(p))

        softplus = nn.Softplus()
        relu = nn.ReLU()

        for i in trange(
            training_options["n_samples"], disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0],
                size=training_options["batch_size"],
                replace=False,
            )
            lr_val = lr_fn(int(i))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_val

            sigma2_ = softplus(sigma2_l) + 1e-3
            psi_ = softplus(psi_l) + 1e-3
            delta_ = softplus(delta_l) + 1e-3

            X_batch = X_[idx]
            y_batch = y_[idx]

            Xb = torch.matmul(X_batch, beta_)
            diff = torch.squeeze(y_batch - Xb)

            f_prior_psi = ptd.Gamma(
                prior_options["a"] * torch.ones(self.p), 1 / delta_ + 0.001
            ).log_prob
            f_prior_sigma = ptd.Gamma(1, 1).log_prob
            f_prior_beta = ptd.Normal(
                zeros_p, torch.sqrt(0.001 + sigma2_ / N * psi_)
            ).log_prob
            f_likelihood = ptd.Normal(0, torch.sqrt(sigma2_)).log_prob

            loglike = torch.mean(f_likelihood(diff))
            prior_psi = torch.sum(f_prior_psi(psi_)) / N
            prior_beta = torch.sum(f_prior_beta(beta_)) / N
            prior_delta = torch.sum(f_prior_delta(delta_)) / N
            prior_sigma = f_prior_sigma(sigma2_) / (2 * N)

            posterior = (
                loglike + prior_beta + prior_psi + prior_delta + prior_sigma
            )
            loss = -1 * posterior + 100 * relu(sigma2_ - 0.3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._store_samples(i, var_list)

    def _fill_prior_options(self, prior_options):
        """ """
        default_options = {"a": 3, "b": 3, "phi": 3}
        return {**default_options, **prior_options}

    def _initialize_variables(self, X, y):
        mod = Ridge()
        mod.fit(X, y)
        y_hat = mod.predict(X)
        mse = np.mean((y - y_hat) ** 2)

        beta_ = torch.tensor(np.squeeze(mod.coef_), requires_grad=True)
        psi_ = torch.ones(X.shape[1], requires_grad=True)
        delta_ = torch.ones(X.shape[1], requires_grad=True)
        sigma2_l = torch.tensor(mse, requires_grad=True)

        return beta_, psi_, delta_, sigma2_l

    def _store_samples(self, i, var_list):
        """
        Saves the learned variables

        Parameters
        ----------
        list_of_samples: list
            List of variables to save
        """
        softplus = nn.Softplus()
        sigma2 = softplus(var_list[3])
        psi = softplus(var_list[1])
        delta = softplus(var_list[2])
        self.samples_beta[i] = var_list[0].detach().numpy()
        self.samples_psi[i] = psi.detach().numpy()
        self.samples_delta[i] = delta.detach().numpy()
        self.samples_sigma2[i] = sigma2.detach().numpy()

    def _test_inputs(self, X, y):
        """
        This performs error checking on inputs for fit

        Parameters
        ----------
        """
        if X.shape[0] != len(y):
            raise ValueError("Sample sizes do not match")

    def _transform_training_data(self, X, y):
        """
        This converts training data to adequate format

        Parameters
        ----------
        """
        X_new = StandardScaler().fit_transform(X)
        y_new = (y - np.mean(y)) / np.std(y)
        X_ = torch.tensor(X_new)
        y_ = torch.tensor(y_new)
        return X_, y_
