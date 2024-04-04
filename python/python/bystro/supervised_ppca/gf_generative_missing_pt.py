"""
This implements standard probabilistic principal component analysis
but with the ability to handle missing data.

Objects
-------
PPCAM(BaseSGDModel)
    Principal component analysis but with Pytorch implementation that 
	can handle missing data

Methods
-------
None
"""
import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.decomposition import PCA
from tqdm import trange
import torch
from torch import Tensor, nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma

from bystro.supervised_ppca._misc_np import (
    softplus_inverse_np,
    classify_missingness,
)
from bystro.supervised_ppca._base import BasePCASGDModel


class PPCAM(BasePCASGDModel):
    """
    This implements PPCA but allowing for missing values
    among the data. This assumes missing completely at
    random, which is reasonable given that our lack of
    measurements likely stems from failure to measure
    rather than some missingness depending on the observed
    covariates. In this case, we simply drop the missing
    parts from the likelihood. Given that PPCA is analytic
    in the marginal likelihood this makes fitting straightforward.
    """

    def __init__(
        self,
        n_components: int = 2,
        prior_options: dict | None = None,
        training_options: dict | None = None,
    ):
        """
                Initializes the model

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
        self._initialize_save_losses()

        self.W_: NDArray[np.float_] | None = None
        self.sigmas_: NDArray[np.float_] | None = None
        self.sigma2_: np.float_ | None = None
        self.p: int | None = None

    def __repr__(self):
        return f"PPCAM(n_components={self.n_components})"

    def fit(
        self,
        X: NDArray[np.float_],
        progress_bar: bool = True,
    ) -> "PPCAM":
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        Returns
        -------
        self : PPCA
            The model
        """
        self._test_inputs(X)
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        W_, sigmal_ = self._initialize_variables(device, X)

        X_list, miss_pat = classify_missingness(X)
        Xt_list = [torch.tensor(X, device=device) for X in X_list]

        n_groups = len(Xt_list)
        p_list = np.array([np.sum(mp) for mp in miss_pat])

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        softplus = nn.Softplus()

        _prior = self._create_prior(device)

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            like_marginal = []
            for y in range(n_groups):
                sigma = Sigma[miss_pat[y]][:, miss_pat[y]]
                mvn = MultivariateNormal(
                    torch.zeros(p_list[y], device=device), sigma
                )
                like_marginal.append(torch.sum(mvn.log_prob(Xt_list[y])))

            like_tot = torch.sum(torch.stack(like_marginal)) / N
            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_tot, like_prior, posterior)

        self._store_instance_variables(device, trainable_variables)

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
        covariance : NDArray(p,p)
            The covariance matrix
        """
        if self.W_ is None or self.sigma2_ is None or self.p is None:
            raise ValueError("Fit model first")

        return np.dot(self.W_.T, self.W_) + self.sigma2_ * np.eye(self.p)

    def get_noise(self):
        """
        Returns the observational noise as a diagonal matrix

        Parameters
        ----------
        None

        Returns
        -------
        Lambda : NDArray,(p,p)
            The observational noise
        """
        if self.sigma2_ is None or self.p is None:
            raise ValueError("Fit model first")

        return self.sigma2_ * np.eye(self.p)

    def _create_prior(self, device):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the log density of the prior
        """
        prior_options = self.prior_options

        def log_prior(trainable_variables: list[Tensor]):
            W_ = trainable_variables[0]
            sigmal_ = trainable_variables[1]
            part1 = (
                -1 * prior_options["weight_W"] * torch.mean(torch.square(W_))
            )
            part2 = (
                Gamma(prior_options["alpha"], prior_options["beta"])
                .log_prob(nn.Softplus()(sigmal_))
                .to(device)
            )
            out = torch.mean(part1 + part2)
            return out

        return log_prior

    def _initialize_variables(
        self, device: Any, X: NDArray[np.float_]
    ) -> tuple[Tensor, Tensor]:
        """
        Initializes the variables of the model. Right now fits a PCA model
        in sklearn, uses the loadings and sets sigma^2 to be unexplained
        variance.

        Parameters
        ----------
        X : NDArray,(n_samples,p)
            The data

        Returns
        -------
        W_ : torch.tensor,shape=(n_components,p)
            The loadings of our latent factor model

        sigmal_ : torch.tensor
            The unrectified variance of the model
        """
        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True, device=device)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = torch.tensor(sinv, requires_grad=True, device=device)
        return W_, sigmal_

    def _save_losses(
        self,
        i: int,
        device: Any,
        log_likelihood: Tensor,
        log_prior: NDArray[np.float_] | Tensor,
        log_posterior,
    ) -> None:
        """
        Saves the values of the losses at each iteration

        Parameters
        -----------
        i : int
            Current training iteration

        losses_likelihood : Tensor
            The log likelihood

        losses_prior : NDArray[np.float_] | Tensor
            The log prior

        losses_posterior : Tensor
            The log posterior
        """
        if device.type == "cuda":
            self.losses_likelihood[i] = log_likelihood.detach().cpu().numpy()
            if isinstance(log_prior, Tensor):
                self.losses_prior[i] = log_prior.detach().cpu().numpy()
            else:
                self.losses_prior[i] = log_prior
            self.losses_posterior[i] = log_posterior.detach().cpu().numpy()
        else:
            self.losses_likelihood[i] = log_likelihood.detach().numpy()
            if isinstance(log_prior, Tensor):
                self.losses_prior[i] = log_prior.detach().numpy()
            else:
                self.losses_prior[i] = log_prior
            self.losses_posterior[i] = log_posterior.detach().numpy()

    def _store_instance_variables(  # type: ignore[override]
        self, device: Any, trainable_variables: list[Tensor]
    ) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
        device ; pytorch.device
            The device used for trainging (gpu or cpu)

        trainable_variables : list[Tensor]
            List of tensorflow variables saved

        Sets
        ----
        W_ : NDArray,(n_components,p)
            The loadings

        sigma2_ : np.float_
            The isotropic variance
        """
        if device.type == "cuda":
            self.W_ = trainable_variables[0].detach().cpu().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().cpu().numpy()
            )
        else:
            self.W_ = trainable_variables[0].detach().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().numpy()
            )

    def _test_inputs(self, X: NDArray[np.float_]) -> None:
        """
        Just tests to make sure data is numpy array
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")

    def _fill_prior_options(
        self, prior_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        default_dict = {"weight_W": 0.01, "alpha": 3.0, "beta": 3.0}

        return {**default_dict, **prior_options}
