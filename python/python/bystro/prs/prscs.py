import numpy as np

from tqdm import trange

import torch
from torch import nn

from sklearn.preprocessing import StandardScalar # type: ignore

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
    def __init__(self, training_options, prior_options):
        self.sample_list = []
        super().__init__(
            training_options=training_options, prior_options=prior_options
        )

    def fit(self, X, y, progress_bar=True, seed=2021):
        """ """
        self._test_inputs(X, y)
        X_, y_ = self._transform_training_data(X, y)
        N, p = X.shape
        self.p = p
        training_options = self.training_options
        prior_options = self.prior_options

        rng = np.random.default_rng(int(seed))

        beta_, psi_, delta_, sigma2_l = self._initialize_variables(X, y)

        self.samples_beta = np.zeros((training_options["n_samples"], self.p))
        self.samples_psi = np.zeros((training_options["n_samples"], self.p))
        self.samples_delta = np.zeros((training_options["n_samples"], self.p))
        self.samples_sigma2 = np.zeros((training_options["n_samples"], self.p))

        var_list = [beta_, psi_, delta_, sigma2_l]

        f_prior_delta = ptd.Gamma(
            prior_options["b"], prior_options["phi"]
        ).log_prob

        lr_fn = scheduler_sgld_geometric(
            n_samples=training_options["n_samples"]
        )
        optimizer = PreconditionedSGLDynamicsPT(
            var_list, lr=0.001, weight_decay=0.0
        )
        zeros_p = torch.tensor(np.zeros(p))
        eye_p = torch.tensor(np.eye(p))

        softplus = nn.Softplus()

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

            sigma2_ = softplus(sigma2_l)

            X_batch = X_[idx]
            y_batch = y_[idx]
            Xb = torch.matmul(X_batch, beta_)
            diff = y_batch - Xb

            f_prior_psi = ptd.Gamma(prior_options["a"], 1 / delta_).log_prob
            f_prior_beta = ptd.Normal(zeros_p, sigma2_ / N * psi_).log_prob
            f_likelihood = ptd.MultivariateNormal(
                zeros_p, covariance_matrix=sigma2_ * eye_p
            ).log_prob

            posterior = (
                f_likelihood(diff)
                + f_prior_psi(psi_)
                + f_prior_beta(beta_)
                + f_prior_delta(delta_)
            )
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._store_samples(i, var_list)

    def _fill_prior_options(self, prior_options):
        """ """
        default_options = {"a": 3, "b": 3, "phi": 3}
        return {**default_options, **prior_options}

    def _store_samples(self, i, var_list):
        """
        Saves the learned variables

        Parameters
        ----------
        list_of_samples: list
            List of variables to save
        """
        softplus = nn.Softplus()
        self.samples_beta[i] = var_list[0].detach().numpy()
        self.samples_psi[i] = var_list[1].detach().numpy()
        self.samples_delta[i] = var_list[2].detach().numpy()
        sigma2 = softplus(var_list[3])
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
        X_new = StandardScalar().fit_transform(X)
        y_new = (y - np.mean(y)) / np.std(y)
        X_ = torch.tensor(X_new)
        y_ = torch.tensor(y_new)
        return X_, y_
