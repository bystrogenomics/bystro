"""
This implements the base class of a model  identifying conditional 
relationships in rare variants. The model is similar to an ising model 
and performs inference using noise contrastive estimation. The model we 
have posited is 

p(x_1,...,x_p) prop exp(Phi_jx_j +  Theta_jk x_jx_k)
Theta_jk >= 0

where x_i is a bernoulli random variable indicating the presence of a 
variant.

Objects
-------
BaseMarkovRandomField(BaseSGDModel)
    This implements a Markov Random Field model of rare variants as an 
    undirected graphical model

Methods
-------
None

"""
from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Any

import torch
from torch import nn
from torch import Tensor

from bystro._template_sgd_np import BaseSGDModel


class BaseMarkovRandomField(BaseSGDModel):
    """
    This implements a Markov Random Field model for rare variant 
    characterization using Noise Contrastive Estimation as an 
    inference method
    """

    def __init__(
        self,
        prior_options: dict | None = None,
        training_options: dict | None = None,
    ):
        """
        This initializes the model

        Parameters
        ----------
        prior_options : 

        training_options : 

        """
        super().__init__(training_options=training_options)
        self.prior_options = self._fill_prior_options(prior_options)

        self.p = 0
        self._initialize_save_losses()
        self.Phi_ = np.empty((2, 2))
        self.Theta_ = np.empty((2, 2))
        self.log_z_ = np.empty((2, 2))

    @abstractmethod
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

    def score(
        self, X: NDArray[np.float_], weights: NDArray[np.float_] | None = None
    ):
        """
        Computes the log likelihood of an array of data
        
        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data

        weights : np.array-like,(N,),default=None
            The optional weights on the samples

        Returns
        -------
        avg_score : float
            Average log likelihood
        """
        w = np.ones(X.shape[0]) if weights is None else weights
        sample_scores = self.score_samples(X)
        avg_score = np.mean(w * sample_scores)
        return avg_score

    def score_samples(self, X):
        """
        Computes the log likelihood of an array of data 
        for each data point
        
        Parameters
        ----------
        X : np.array-like,(N,sum(p))
            The data

        Returns
        -------
        llike: float
            The per sample log likelihood
        """
        XT = np.dot(X, self.Theta_)
        XTX = X * XT
        llike = np.sum(XTX, axis=0) + self.log_z_
        return llike

    def _initialize_variables(
        self, X: NDArray[np.float_]
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        Phi_ : torch.tensor,shape=(p)
            The latent activations

        L_l: torch.tensor
            The conditional latent parameters

        log_z_ : torch.tensor
            The estimate of the normalization constant
        """
        p_marginal = np.mean(X, axis=0)
        Phi_ = torch.tensor(np.log(p_marginal) - 1.0, requires_grad=True)
        L_l = torch.tensor(
            1 / self.p * np.ones((self.p, self.p)).astype(np.float32),
            requires_grad=True,
        )
        log_z_ = torch.tensor(0.0, requires_grad=True)

        return Phi_, L_l, log_z_

    def _initialize_save_losses(self) -> None:
        """
        This method initializes the arrays to track relevant variables
        during training at each iteration

        Sets
        ----
        losses_prediction : np.array(n_iterations)
            The log likelihood

        """
        n_iterations = self.training_options["n_iterations"]
        self.losses_prediction = np.empty(n_iterations)

    def _save_losses(self, i, loss_predict: Tensor,) -> None:
        """
        Saves the values of the losses at each iteration

        Parameters
        -----------
        i : int
            Current training iteration

        loss_predict : Tensor
            The predictive loss
        """
        self.losses_prediction[i] = loss_predict.detach().numpy()

    def _store_instance_variables(
        self, trainable_variables: list[Tensor]
    ) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list[Tensor]
            List of tensorflow variables saved

        Sets
        ----
        """
        self.Phi_ = trainable_variables[0].detach().numpy()
        L = torch.tril(nn.ReLU()(trainable_variables[1]), diagonal=-1)
        Theta_ = 0.5 * (L + torch.transpose(L, 0, 1))
        self.Theta_ = Theta_.detach().numpy()
        self.log_z_ = trainable_variables[2].detach().numpy()

    def _test_inputs(self, X: NDArray[np.float_]) -> None:
        """
        Just tests to make sure data is numpy array
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
        list_unique = np.unique(X)
        if len(list_unique) != 2:
            print("Valid values are 0 and 1")

    def _transform_training_data(self, *args: NDArray) -> list[Tensor]:
        """ 
        Convert a list of numpy arrays to tensors
        """
        out = []
        for arg in args:
            out.append(torch.tensor(arg))
        return out

    def _fill_prior_options(self, prior_options):
        """
        Fills in options for prior parameters

        Paramters
        ---------
        new_dict : dictionary
            The prior parameters used to specify the prior
        """
        default_dict = {
            "mu_phi": -2.0,
            "sigma_phi": 1.0,
            "mu_L_l": -1.0,
            "sigma_L_l": 1.0,
        }
        new_dict = {**default_dict, **prior_options}
        return new_dict

    def _fill_training_options(
        self, training_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        This sets the default parameters for stochastic gradient descent,
        our inference strategy for the model.

        Parameters
        ----------
        training_options : dict
            The original options set by the user passed as a dictionary

        Options
        -------
        n_iterations : int, default=3000
            Number of iterations to train using stochastic gradient descent

        learning_rate : float, default=5e-4
            Learning rate of gradient descent

        batch_size : int, default=1000
            The number of observations to use at each iteration. If none
            corresponds to batch learning

        nu : int,default=10
            The relative proportion of noise samples to real samples
        """
        default_options = {
            "n_iterations": 3000,
            "learning_rate": 5e-4,
            "batch_size": 100,
            "nu": 10,
        }
        tops = {**default_options, **training_options}

        default_keys = set(default_options.keys())
        final_keys = set(tops.keys())

        expected_but_missing_keys = default_keys - final_keys
        unexpected_but_present_keys = final_keys - default_keys
        if expected_but_missing_keys:
            raise ValueError("Training keys missing")
        if unexpected_but_present_keys:
            raise ValueError("Unrecognized training keys given")

        return tops
