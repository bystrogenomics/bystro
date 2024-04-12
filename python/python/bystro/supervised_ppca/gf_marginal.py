import numpy as np
from numpy.typing import NDArray
from typing import List, Optional

from tqdm import trange
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from bystro.supervised_ppca.gf_generative_pt import PPCA
from bystro.supervised_ppca._base import (
    _get_projection_matrix,
    kl_divergence_vae,
)


class BaseMarginal(PPCA):
    def _test_inputs(  # type: ignore[override]
        self,
        X: NDArray[np.float_],
        idx_list: List[NDArray[np.bool_]],
        lamb: Optional[NDArray[np.float_]],
    ) -> None:
        """
        Validate the input parameters.

        Parameters
        ----------
        X : NDArray[np.float_]
            The input data array.
        idx_list : List[NDArray[np.bool_]]
            List of boolean arrays indicating observed dimensions.
        lamb : Optional[NDArray[np.float_]]
            Array of regularization parameters for each group.

        Raises
        ------
        ValueError
            If input parameters are invalid, such as mismatched dimensions.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data must be numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
        for idx_array in idx_list:
            if idx_array.dtype != np.bool_:
                raise ValueError(
                    "idx_list must be a list of NumPy boolean arrays"
                )
        if lamb is not None and not isinstance(lamb, np.ndarray):
            raise ValueError("lamb must be a NumPy array if it's not None")
        if lamb is not None and len(lamb) != len(idx_list):
            raise ValueError(
                "Mismatch in lamb length and number of groups in idx_list"
            )


class PPCAMarginal(BaseMarginal):
    """
    Probabilistic Principal Component Analysis (PPCA) with marginal
    supervision.

    This class extends PPCA to handle cases where certain dimensions of
    the data are missing. It weights the marginal predictive distribution
    integrating out the latent variables to perform supervision. This
    yields a model which is more predictive of the missing data.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    prior_options : dict or None, optional, default=None
        Dictionary specifying the prior options. Currently not
        used in this implementation.
    gamma : float, default=10.0
        Parameter to ensure independence of factors
    training_options : dict or None, optional, default=None
        Dictionary specifying training options. Keys can include
        'learning_rate' and 'momentum' for the SGD optimizer.

    Attributes
    ----------
    gamma : float
        Regularization parameter.
    """

    def __init__(
        self,
        n_components: int = 2,
        prior_options: dict | None = None,
        gamma: float = 10.0,
        training_options: dict | None = None,
    ) -> None:
        self.gamma: float = float(gamma)
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self.losses_pred = np.zeros(self.training_options["n_iterations"])
        self._initialize_save_losses()

    # override needed for mypy to ignore the non-optional `y` argument
    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float_],
        idx_list: List[NDArray[np.bool_]],
        lamb: NDArray[np.float_],
        progress_bar: bool = True,
        seed: int = 2021,
    ) -> "PPCAMarginal":
        """
        Fit the PPCA model with marginalization to the given data.

        Parameters
        ----------
        X : NDArray[np.float_]
            The input data array. Must be 2-dimensional.
        idx_list : List[NDArray[np.bool_]]
            List of boolean arrays where True indicates observed dimensions and
            False indicates dimensions to marginalize over.
        lamb : Optional[NDArray[np.float_]], default=None
            Array of regularization parameters for each group in idx_list. If
            None, uses uniform regularization.
        progress_bar : bool, default=True
            Whether to display a progress bar during training.
        seed : int, default=2021
            Seed for the random number generator.

        Returns
        -------
        self : PPCAMarginal
            The fitted model.
        """
        self._test_inputs(X, idx_list, lamb)
        rng = np.random.default_rng(int(seed))
        training_options = self.training_options
        N, p = X.shape
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        n_groups = len(idx_list)
        self.idx_list = idx_list
        self.n_groups, self.p = n_groups, p
        n_g_indiv = np.zeros(n_groups)
        vals = np.arange(p)
        for i in range(n_groups):
            n_g_indiv[i] = np.sum(1 * idx_list[i])
        n_g_indiv = n_g_indiv.astype(int)
        idxs = [
            torch.tensor(vals[idx_list[i]], device=device)
            for i in range(n_groups)
        ]
        idxs_c = [
            torch.tensor(vals[~idx_list[i]], device=device)
            for i in range(n_groups)
        ]

        W_, sigmal_ = self._initialize_variables(device, X)
        X_ = self._transform_training_data(device, X)[0]

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        softplus = nn.Softplus()

        _prior = self._create_prior(device)
        z_vecs = [
            torch.zeros(n_g_indiv[i], device=device) for i in range(n_groups)
        ]

        Lamb = torch.tensor(lamb, device=device)

        for i in trange(
            int(training_options["n_iterations"]), disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]

            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            like_prior = _prior(trainable_variables)

            like_gens = []
            like_preds = []

            for k in range(n_groups):
                Sigma_marg = Sigma[torch.meshgrid(idxs[k], idxs[k])]
                X_o = X_batch[:, idxs[k]]
                X_m = X_batch[:, idxs_c[k]]
                m = MultivariateNormal(z_vecs[k], Sigma_marg)
                like_gens.append(torch.mean(m.log_prob(X_o)))

                Sigma_21 = Sigma[torch.meshgrid(idxs_c[k], idxs[k])]
                Sigma_11 = Sigma[torch.meshgrid(idxs_c[k], idxs_c[k])]

                X_o_s = torch.linalg.solve(
                    Sigma_marg, torch.transpose(X_o, 0, 1)
                )

                projection = torch.matmul(Sigma_21, X_o_s)

                X_bar = X_m - torch.transpose(projection, 0, 1)
                m2 = MultivariateNormal(
                    torch.zeros(p - n_g_indiv[k], device=device), Sigma_11
                )
                like_preds.append(Lamb[k] * torch.mean(m2.log_prob(X_bar)))

            WTW = torch.matmul(W_, torch.transpose(W_, 0, 1))
            off_diag = WTW - torch.diag(torch.diag(WTW))
            loss_i = torch.linalg.matrix_norm(off_diag)

            like_gen = torch.sum(torch.stack(like_gens))
            like_pred = torch.sum(torch.stack(like_preds))
            self.losses_pred[i] = like_pred.detach().cpu().numpy()

            posterior = like_gen + like_pred + 1 / N * like_prior
            loss = -1 * posterior + self.gamma * loss_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_gen, like_prior, posterior)

        self._store_instance_variables(device, trainable_variables)

        return self


class PPCAMarginalKL(BaseMarginal):
    """
    Probabilistic Principal Component Analysis (PPCA) with marginal
    supervision.

    This class extends PPCA to handle cases where certain dimensions of
    the data are missing. It weights the marginal predictive distribution
    integrating out the latent variables to perform supervision. This
    yields a model which is more predictive of the missing data.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    prior_options : dict or None, optional, default=None
        Dictionary specifying the prior options. Currently not
        used in this implementation.
    gamma : float, default=10.0
        Parameter to ensure independence of factors
    training_options : dict or None, optional, default=None
        Dictionary specifying training options. Keys can include
        'learning_rate' and 'momentum' for the SGD optimizer.

    Attributes
    ----------
    gamma : float
        Regularization parameter.
    """

    def __init__(
        self,
        n_components: int = 2,
        prior_options: dict | None = None,
        gamma: float = 10.0,
        training_options: dict | None = None,
    ) -> None:
        self.gamma: float = float(gamma)
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self.losses_pred = np.zeros(self.training_options["n_iterations"])
        self._initialize_save_losses()

    # override needed for mypy to ignore the non-optional `y` argument
    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float_],
        idx_list: List[NDArray[np.bool_]],
        lamb: NDArray[np.float_],
        progress_bar: bool = True,
        seed: int = 2021,
    ) -> "PPCAMarginalKL":
        """
        Fit the PPCA model with marginalization to the given data.

        Parameters
        ----------
        X : NDArray[np.float_]
            The input data array. Must be 2-dimensional.
        idx_list : List[NDArray[np.bool_]]
            List of boolean arrays where True indicates observed dimensions and
            False indicates dimensions to marginalize over.
        lamb : Optional[NDArray[np.float_]], default=None
            Array of regularization parameters for each group in idx_list. If
            None, uses uniform regularization.
        progress_bar : bool, default=True
            Whether to display a progress bar during training.
        seed : int, default=2021
            Seed for the random number generator.

        Returns
        -------
        self : PPCAMarginal
            The fitted model.
        """
        self._test_inputs(X, idx_list, lamb)
        rng = np.random.default_rng(int(seed))
        training_options = self.training_options
        N, p = X.shape
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        n_groups = len(idx_list)
        self.idx_list = idx_list
        self.n_groups, self.p = n_groups, p
        n_g_indiv = np.zeros(n_groups)
        vals = np.arange(p)
        for i in range(n_groups):
            n_g_indiv[i] = np.sum(1 * idx_list[i])
        n_g_indiv = n_g_indiv.astype(int)
        idxs = [
            torch.tensor(vals[idx_list[i]], device=device)
            for i in range(n_groups)
        ]
        idxs_c = [
            torch.tensor(vals[~idx_list[i]], device=device)
            for i in range(n_groups)
        ]

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
        z_vecs = [
            torch.zeros(n_g_indiv[i], device=device) for i in range(n_groups)
        ]
        z_vecs_c = [
            torch.zeros(p - n_g_indiv[i], device=device)
            for i in range(n_groups)
        ]
        eyes = [torch.eye(n_g_indiv[i], device=device) for i in range(n_groups)]
        eyes_c = [
            torch.eye(p - n_g_indiv[i], device=device) for i in range(n_groups)
        ]

        Lamb = torch.tensor(lamb, device=device)

        for i in trange(
            int(training_options["n_iterations"]), disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]
            sigma = softplus(sigmal_)

            # Perform reconstruction
            P_x, Cov = _get_projection_matrix(W_, sigma, device)
            mean_z = torch.matmul(X_batch, torch.transpose(P_x, 0, 1))
            eps = torch.rand_like(mean_z)
            C1_2 = torch.linalg.cholesky(Cov)
            z_samples = mean_z + torch.matmul(eps, C1_2)
            X_recon = torch.matmul(z_samples, W_)
            X_diff = X_batch - X_recon

            like_prior = _prior(trainable_variables)
            like_gen_kl = torch.mean(kl_divergence_vae(mean_z, Cov))

            like_gens = []
            like_preds = []

            for k in range(n_groups):
                X_o = X_diff[:, idxs[k]]
                X_m = X_diff[:, idxs_c[k]]
                m1 = MultivariateNormal(z_vecs[k], sigma * eyes[k])
                m2 = MultivariateNormal(z_vecs_c[k], sigma * eyes_c[k])
                like_gens.append(torch.mean(m1.log_prob(X_o)))
                like_preds.append(Lamb[k] * torch.mean(m2.log_prob(X_m)))

            WTW = torch.matmul(W_, torch.transpose(W_, 0, 1))
            off_diag = WTW - torch.diag(torch.diag(WTW))
            loss_i = torch.linalg.matrix_norm(off_diag)

            like_gen = torch.sum(torch.stack(like_gens))
            like_pred = torch.sum(torch.stack(like_preds))
            self.losses_pred[i] = like_pred.detach().cpu().numpy()

            posterior = like_gen + like_pred + 1 / N * like_prior - like_gen_kl
            loss = -1 * posterior + self.gamma * loss_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_gen, like_prior, posterior)

        self._store_instance_variables(device, trainable_variables)

        return self
