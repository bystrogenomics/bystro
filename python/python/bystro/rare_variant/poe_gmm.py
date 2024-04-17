"""
This module provides an implementation for detecting parent of origin 
effects in genetic data. It utilizes a Gaussian Mixture Model (GMM) 
approach with fixed mixture weights and covariance to model the effects 
of alleles depending on whether they were inherited from the mother or 
the father. The `POEGMM` class extends a base stochastic gradient descent 
model to incorporate this specialized fitting process.
"""
import numpy as np
from numpy.typing import NDArray
import numpy.linalg as la
from typing import Any

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from tqdm import trange
from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn.mixture import GaussianMixture

from bystro._template_sgd_np import BaseSGDModel


class POEGMM(BaseSGDModel):
    """
    A class to detect parent of origin effects using a Gaussian Mixture Model.

    This class extends a base stochastic gradient descent model to fit a
    GMM with fixed mixture weights and covariance, tailored for genetic
    data to discern the effects based on parental origin.

    Parameters
    ----------
    mu : float, default=0.1
        The strength of sparsity penalty
    covariance_estimation : str, default="Empirical"
        The method used for estimating the covariance. Currently,
        only "Empirical" is supported.
    loss_function : str, default="l1"
        The loss function to use for model fitting. Can be "l1" or "l2".
    training_options : dict, optional
        A dictionary containing options for training the model.
    prior_options : dict, optional
        A dictionary containing options for the prior
        distribution settings.

    Attributes
    ----------
    Sigma_AA_ : NDArray[np.float_]
        The estimated covariance matrix for homozygotes.
    parent_effect_ : NDArray[np.float_]
        The estimated effect sizes attributed to parent of origin.
    effect_ : NDArray[np.float_]
        The general effect sizes estimated by the model.
    """

    def __init__(
        self,
        mu=0.1,
        covariance_estimation="Empirical",
        loss_function="l1",
        training_options=None,
        prior_options=None,
    ) -> None:
        super().__init__(training_options=training_options)
        self._initialize_save_losses()

        if prior_options is None:
            prior_options = {}
        self.prior_options = self._fill_prior_options(prior_options)

        self.covariance_estimation = covariance_estimation
        self.loss_function = loss_function
        self.mu = mu

        self.Sigma_AA_: NDArray[np.float_] | None = None
        self.parent_effect_: NDArray[np.float_] | None = None
        self.effect_: NDArray[np.float_] | None = None

    def fit(self, phenotype, genotype, progress_bar=True, seed=2021):
        """
        Fits the Gaussian Mixture Model to the genetic data to
        detect parent of origin effects.

        Parameters
        ----------
        phenotype : ndarray
            The phenotype data as a numpy array.

        genotype : ndarray
            The genotype data as a numpy array, indicating
            homozygotes and heterozygotes.

        progress_bar : bool, default=True
            If True, displays a progress bar during the model
            fitting process.

        seed : int, default=2021
            The random seed for initialization, ensuring reproducibility.

        Returns
        -------
        self : object
            The instance itself.
        """
        training_options = self.training_options
        self._test_inputs(phenotype, genotype)
        self.n_phenotypes = phenotype.shape[1]

        homozygotes = phenotype[genotype == 0]
        heterozygotes = phenotype[genotype == 1]

        self.effect_ = np.mean(heterozygotes, axis=0) - np.mean(
            homozygotes, axis=0
        )

        homozygotes_dm = homozygotes - np.mean(homozygotes, axis=0)
        heterozygotes_dm = heterozygotes - np.mean(heterozygotes, axis=0)

        Sigma_AA = self._estimate_homozygote_covariance(homozygotes_dm)
        L = la.cholesky(Sigma_AA * 0.95 + 0.05 * np.eye(self.n_phenotypes))
        L_inv = la.inv(L)

        heterozygotes_white = np.dot(heterozygotes_dm, L_inv.T)

        beta_ = self._initialize_variables(heterozygotes_white)
        trainable_variables = [beta_]

        Xw = torch.tensor(heterozygotes_white)

        cov_t = torch.tensor(np.eye(self.n_phenotypes).astype(np.float32))

        loss_fn: _Loss
        if self.loss_function == "l1":
            loss_fn = nn.L1Loss()
        elif self.loss_function == "l2":
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unrecognized loss: %s" % self.loss_function)

        self.Sigma_AA_ = Sigma_AA

        rng = np.random.default_rng(seed)

        _prior = self._create_prior()

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        print(L[:5, :5])

        weight = torch.tensor(0.5)
        Lt = torch.tensor(L.T)

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            if training_options["batch_size"] is None:
                X_batch = Xw.float()
            else:
                idx = rng.choice(
                    Xw.shape[0],
                    size=training_options["batch_size"],
                    replace=False,
                )
                X_batch = Xw[idx].float()

            m1 = MultivariateNormal(beta_, cov_t)
            m2 = MultivariateNormal(-1 * beta_, cov_t)

            loss1 = m1.log_prob(X_batch) + torch.log(weight)
            loss2 = m2.log_prob(X_batch) + torch.log(weight)
            LL = torch.stack((loss1, loss2))
            l2 = torch.logsumexp(LL, dim=0)
            loss_recon = torch.mean(-1 * l2)

            beta_trans = torch.matmul(beta_, Lt)

            loss_reg = loss_fn(beta_trans, torch.zeros_like(beta_))

            loss = loss_recon + self.mu * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, loss_recon, loss_reg)

        self._store_instance_variables(trainable_variables)

        return self

    def transform(self, X, return_inner=False):
        """
        Applies the learned model to predict parent of origin based on
        phenotype.

        Parameters
        ----------
        X : ndarray
            New data to transform.

        return_inner : bool, default=False
            If True, returns the inner product of the transformation
            along with the transformation itself.

        Returns
        -------
        calls : ndarray
            The transformed data.

        preds : ndarray, optional
            The inner product of the transformation, returned
            if `return_inner` is True.
        """
        if self.parent_effect_ is None:
            raise ValueError("parent_effect_ has not been initialized.")
        X_dm = X - np.mean(X, axis=0)
        preds = np.dot(X_dm, self.parent_effect_)
        calls = 1.0 * (preds > 0)
        if return_inner is False:
            return calls
        return calls, preds

    def _create_prior(self):
        """
        Creates a prior function based on the prior options provided
        during initialization.

        The prior function is determined by the `penalty` option in
        `prior_options`. Currently, only an L1 penalty is supported.

        Returns
        -------
        prior_fn : function
            The prior function to be applied on the model parameters.
        """
        prior_options = self.prior_options

        if prior_options["penalty"] == "l1":
            prior_fn = nn.L1Loss()
        else:
            raise ValueError("unrecognized prior option")

        return prior_fn

    def _estimate_homozygote_covariance(self, phenotype):
        """
        Estimates the covariance matrix for homozygotes.

        The method for covariance estimation is specified by
        `covariance_estimation`. Currently, only "Empirical" estimation
        is supported.

        Parameters
        ----------
        phenotype : ndarray
            The phenotype data for homozygotes, used for covariance
            estimation.

        Returns
        -------
        Sigma_AA : ndarray
            The estimated covariance matrix for homozygotes.
        """
        if self.covariance_estimation == "Empirical":
            Sigma_AA = np.cov(phenotype.T)
        else:
            raise ValueError(
                "Unrecognized option %s", self.covariance_estimation
            )

        return Sigma_AA

    def _fill_prior_options(
        self, prior_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fills in the prior options with defaults if not provided.

        Parameters
        ----------
        prior_options : dict[str, Any]
            The prior options provided by the user.

        Returns
        -------
        filled_prior_options : dict[str, Any]
            The prior options filled with default values for missing parameters.
        """
        default_dict = {"penalty": "l1"}

        return {**default_dict, **prior_options}

    def _fill_training_options(self, training_options):
        default_dict = {
            "n_iterations": 10000,
            "learning_rate": 1e-3,
            "batch_size": None,
            "momentum": 0.95,
        }

        return {**default_dict, **training_options}

    def _store_instance_variables(self, trainable_variables):
        """
        Stores the instance variables after training.

        Parameters
        ----------
        trainable_variables : list
            The list of trainable variables, including the
            estimated parent effect.
        """
        if self.Sigma_AA_ is None:
            raise ValueError("Sigma_AA_ has not been initialized.")
        parent_effect_w = np.squeeze(trainable_variables[0].detach().numpy())
        L = la.cholesky(self.Sigma_AA_)
        self.parent_effect_ = np.dot(parent_effect_w, L.T) * 2

    def _initialize_save_losses(self):
        """
        Initializes the arrays to save losses during training.
        """
        self.losses_likelihoods = np.zeros(
            self.training_options["n_iterations"]
        )
        self.losses_regularization = np.zeros(
            self.training_options["n_iterations"]
        )

    def _initialize_variables(self, heterozygotes_white):
        """
        Initializes the variables for the optimization process.

        Parameters
        ----------
        heterozygotes_white : ndarray
            The whitened data for heterozygotes.

        Returns
        -------
        beta_ : Tensor
            The initial values for the optimization variables.
        """
        mod = GaussianMixture(2)
        mod.fit(heterozygotes_white)
        beta_i = (mod.means_[0] - mod.means_[1]) / 2
        beta_ = torch.tensor(beta_i, requires_grad=True)
        return beta_

    def _save_losses(self, i, loss_recon, loss_regularization):
        """
        Saves the reconstruction and regularization losses at iteration i.

        Parameters
        ----------
        i : int
            The current iteration.
        loss_recon : Tensor
            The reconstruction loss.
        loss_regularization : Tensor
            The regularization loss.
        """
        self.losses_likelihoods[i] = loss_recon.detach().numpy()
        self.losses_regularization[i] = loss_regularization.detach().numpy()

    def _transform_training_data(self, *args, **kwargs):
        """
        Placeholder required by template
        """

    def _test_inputs(self, phenotype, genotype):
        """
        Tests the inputs for compatibility with the model requirements.

        Parameters
        ----------
        phenotype : ndarray
            The phenotype data.
        genotype : ndarray
            The genotype data.

        Raises
        ------
        ValueError
            If the inputs do not meet the required format or dimensions.
        """
        if not isinstance(phenotype, np.ndarray):
            raise ValueError("phenotype is numpy array")
        if not isinstance(genotype, np.ndarray):
            raise ValueError("genotype is numpy array")
        if phenotype.shape[0] != len(genotype):
            raise ValueError("phenotype and genotype have different samples")
