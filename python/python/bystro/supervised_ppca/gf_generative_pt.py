"""
This implements Gaussian factor analysis models with generative (standard)
inference. There are three different models. Gaussian factor analysis 
parameterizes the covariance matrix as 

    Sigma = WW^T + Lambda

where Lambda is a diagonal matrix (see Bishop 2006 Chapter 12 for notation
and model definition).  Probabilistic principal component analysis 
sets Lambda = sigma**2*I_p. Supervised principal component analysis 
parameterizes it as 
Lambda = diag([sigma_x**2*1_p,sigma_y**2 1_q]), allowing for variances to 
differ between the predictive and dependent variables. Finally, factor 
analysis allows each diagonal component to be distinct. Models (1) and (3)
are described in Bishop 2006, while supervised Probabilistic PCA is 
described in several papers, including Yu 2006.

Note that in the code W is a L x p array, where L is the latent dimensionality 
and  p is the covariate dimension, for implementation convenience. However, 
mathematically it is often pxL for notational convenience. Given that the 
most insidious errors are mathematical in nature rather than coding, (as faulty
math is difficult to detect in unit tests), our notation matches mathematics 
rather than code, specifically when naming WWT and WTW to match the Bishop 2006
notation rather than code.

Objects
-------
PPCApt(BaseSGDModel)
    Principal component analysis but with Pytorch implementation.

FactorAnalysispt(BaseSGDModel)
    Factor analysis implemented in pytorch. See Bishop 2006

Methods
-------
None
"""
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable

from sklearn.decomposition import PCA
from tqdm import trange
import torch
from torch import Tensor, nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma

from bystro.supervised_ppca._misc_np import softplus_inverse_np
from bystro.supervised_ppca._base import BasePCASGDModel
from bystro.supervised_ppca._sherman_woodbury_pt import mvn_log_prob_sw


class PPCA(BasePCASGDModel):
    def __init__(
        self,
        n_components: int = 2,
        prior_options: dict | None = None,
        training_options: dict | None = None,
    ):
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
        self._initialize_save_losses()

        self.W_: NDArray[np.float_] | None = None
        self.sigmas_: NDArray[np.float_] | None = None
        self.sigma2_: np.float_ | None = None
        self.p: int | None = None

    def __repr__(self):
        return f"PPCApt(n_components={self.n_components})"

    def fit(
        self,
        X: NDArray[np.float_],
        progress_bar: bool = True,
        seed: int = 2021,
        sherman_woodbury: bool = False,
    ) -> "PPCA":
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        seed : int,default=2021
            The seed of the random number generator

        sherman_woodbury : bool,default=False
            Whether to use the Sherman Woodbury identity to calculate
            the likelihood. Advantageous in high-p situations

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

        rng = np.random.default_rng(int(seed))

        W_, sigmal_ = self._initialize_variables(device, X)

        X_tensor = self._transform_training_data(device, X)[0]

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        eye_L = torch.tensor(
            np.eye(self.n_components).astype(np.float32), device=device
        )
        zeros_p = torch.tensor(
            np.zeros(p).astype(np.float32), dtype=torch.float32, device=device
        )
        softplus = nn.Softplus()

        _prior = self._create_prior(device)

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            idx = rng.choice(
                X_tensor.shape[0],
                size=training_options["batch_size"],
                replace=False,
            )
            X_batch = X_tensor[idx].float()

            sigma = softplus(sigmal_)

            if sherman_woodbury:
                Lambda = sigma * eye
                like_tot = mvn_log_prob_sw(X_batch, zeros_p, Lambda, W_, eye_L)
            else:
                WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
                Sigma = WWT + sigma * eye
                m = MultivariateNormal(zeros_p, Sigma)
                like_tot = torch.mean(m.log_prob(X_batch))

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
            sigma_ = nn.Softplus()(sigmal_)

            part1 = (
                -1 * prior_options["weight_W"] * torch.mean(torch.square(W_))
            )
            part2 = (
                Gamma(prior_options["alpha"], prior_options["beta"])
                .log_prob(sigma_)
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
        device ; pytorch.device
            The device used for trainging (gpu or cpu)

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

    def _store_instance_variables(  # type: ignore[override]
        self, device: Any, trainable_variables: list[Tensor]
    ) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
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
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")

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


class FactorAnalysis(BasePCASGDModel):
    def __init__(
        self,
        n_components=2,
        prior_options: dict | None = None,
        training_options: dict | None = None,
    ):
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
        self._initialize_save_losses()
        self.p: int | None = None
        self.W_: NDArray[np.float_] | None = None
        self.sigmas_: NDArray[np.float_] | None = None

    def __repr__(self) -> str:
        return f"FactorAnalysispt(n_components={self.n_components})"

    def fit(
        self,
        X: NDArray[np.float_],
        progress_bar: bool = True,
        seed: int = 2021,
        sherman_woodbury: bool = False,
    ) -> "FactorAnalysis":
        """
        Fits a model given covariates X

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        Returns
        -------
        self : FactorAnalysis
            The model
        """
        self._test_inputs(X)
        training_options = self.training_options
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        N, p = X.shape
        self.p = p

        rng = np.random.default_rng(int(seed))

        W_, sigmal_ = self._initialize_variables(device, X)

        X_ = self._transform_training_data(device, X)[0]

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        softplus = nn.Softplus()

        _prior = self._create_prior(device)

        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        zeros_p = torch.zeros(p, device=device)

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]

            sigmas = softplus(sigmal_)
            Lambda = torch.diag(sigmas)

            if sherman_woodbury:
                like_tot = mvn_log_prob_sw(X_batch, zeros_p, Lambda, W_, eye)
            else:
                WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
                Sigma = WWT + Lambda
                m = MultivariateNormal(zeros_p, Sigma)
                like_tot = torch.mean(m.log_prob(X_batch))
            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_tot, like_prior, posterior)

        self._store_instance_variables(device, trainable_variables)

        return self

    def get_covariance(self) -> NDArray[np.float_]:
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
        if self.W_ is None or self.sigmas_ is None:
            raise ValueError("Fit model first")

        return np.dot(self.W_.T, self.W_) + np.diag(self.sigmas_)

    def get_noise(self) -> NDArray[np.float_]:
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
        if self.sigmas_ is None:
            raise ValueError("Fit model first")

        return np.diag(self.sigmas_)

    def _create_prior(self, device) -> Callable[[list[Tensor]], Tensor]:
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the negative log density of the prior
        """

        def log_prior(trainable_variables: list[Tensor]) -> Tensor:
            sigma_ = nn.Softmax()(trainable_variables[1])
            return torch.mean(
                Gamma(self.prior_options["alpha"], self.prior_options["beta"])
                .log_prob(sigma_)
                .to(device)
            )

        return log_prior

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
        default_dict = {"alpha": 3.0, "beta": 3.0}
        return {**default_dict, **prior_options}

    def _initialize_variables(self, device: Any, X: NDArray[np.float_]):
        """
        Initializes the variables of the model by fitting PCA model in
        sklearn and using those loadings

        Parameters
        ----------
        X : NDArray,(n_samples,p)
            The data

        Returns
        -------
        W_ : torch.tensor,(n_components,p)
            The loadings of our latent factor model

        sigmal_ : torch.tensor,(p,)
            The noise of each covariate, unrectified
        """
        if self.p is None:
            raise ValueError("Fit model first")

        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True, device=device)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = torch.tensor(
            sinv[0] * np.ones(self.p).astype(np.float32),
            requires_grad=True,
            device=device,
        )

        return W_, sigmal_

    def _store_instance_variables(  # type: ignore[override]
        self, device: Any, trainable_variables: list[Tensor]
    ) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of tensorflow variables saved

        Sets
        ----
        W_ : NDArray,(n_components,p)
            The loadings

        sigmas_ : NDArray,(n_components,p)
            The diagonal variances
        """
        if device.type == "cuda":
            self.W_ = trainable_variables[0].detach().cpu().numpy()
            self.sigmas_ = (
                nn.Softplus()(trainable_variables[1]).detach().cpu().numpy()
            )
        else:
            self.W_ = trainable_variables[0].detach().numpy()
            self.sigmas_ = (
                nn.Softplus()(trainable_variables[1]).detach().numpy()
            )

    def _test_inputs(self, X: NDArray[np.float_]):
        """
        Just tests to make sure data is numpy array
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
