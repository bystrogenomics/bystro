"""
This is a method for estimating local ancestry as a competitor to G nomix and 
RFMix. This method is a multi-task neural network for predicting local ancestry
in contrast to the random forest-based alternatives. Like RF-mix and its 
extensions, each segment has it's own predictive model for predicting the 
ancestry but is a neural network instead. The network is relatively simplistic,
a multi-layer perceptron with dropout for regularization. The unique aspect of 
this model as compared to the others is that all but the first layer of each
model is shared between every predictive model.

There are two major benefits, provided this model provides even reasonably
close accuracy to Gnomix and RFmix. First, this is completely stochastic in 
sample size, meaning that it will be equally quick to train on large datasets
with hundreds of thousands of individuals just as quickly as hundreds. It is
also nearly-stochastic in the number of regions considered. Most of the layers
are shared with only the initial layer being unique. As such, the shared layers
can be trained stochastically on the different predictive regions. Once these 
layers are finalized, training each individual classifier only requires fitting
the initial layer which should only take a small number of iterations.

The second benefit is that this setup with shared layers should provide us with
substantial improvements in model accuracy. If not, it is most likely due to 
poor architecture. The reason is that multi-task learning has been enormously
successful at regularizing complex neural networks. Here, however, we are taking
different input data to make an identical predictive task unlike the traditional
use of identical data to make different output predictions. Could certainly add
that as well in the future, should the initial approach prove promising.

Objects
-------

Methods
-------

"""
from typing import Any
from abc import ABC
import numpy as np


import torch
from torch import nn, optim
import torch.nn.functional as f

from tqdm import trange

from bystro._template_sgd_np import BaseSGDModel


class BaseMultiAncestry(BaseSGDModel, ABC):
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

        learning_rate : float, default=1e-4
            Learning rate of gradient descent

        method : string {'Nadam'}, default='Nadam'
            The learning algorithm

        batch_size : int, default=None
            The number of observations to use at each iteration. If none
            corresponds to batch learning
        """
        default_options = {
            "learning_rate": 1e-3,
            "n_iterations": 500,
            "n_inital_iterations": 100,
            "n_final_iterations": 100,
            "n_epochs": 50,
            "batch_size": 100,
            "bs_region": 10,
        }
        tops = {**default_options, **training_options}

        default_keys = set(default_options.keys())
        final_keys = set(tops.keys())

        expected_but_missing_keys = default_keys - final_keys
        unexpected_but_present_keys = final_keys - default_keys
        if expected_but_missing_keys:
            raise ValueError("training options were expected but not found...")
        if unexpected_but_present_keys:
            raise ValueError(
                "training options were unrecognized but provided..."
            )
        return tops

    def _initialize_save_losses(self) -> None:
        """
        This method initializes the arrays to track relevant variables
        during training at each iteration

        Sets
        ----
        losses_likelihood : np.array(n_iterations)
            The log likelihood

        losses_prior : np.array(n_iterations)
            The log prior

        losses_posterior : np.array(n_iterations)
            The log posterior
        """
        n_iterations = self.training_options["n_iterations"]
        self.losses_likelihood = np.empty(n_iterations)
        self.losses_prior = np.empty(n_iterations)
        self.losses_posterior = np.empty(n_iterations)

    def _store_instance_variables(self, trainable_variables) -> None:
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list[Tensor]
            List of tensorflow variables saved

        Sets
        ----
        """
        self.shared_model = trainable_variables[0]
        self.individual_models = trainable_variables[1]

    def _test_inputs(self, list_data, ancestry):
        """
        Just tests to make sure data is proper
        dimensionality
        """
        n_samples = len(ancestry)
        n_groups = len(list_data)
        for i in range(n_groups):
            if list_data[i].shape[0] != n_samples:
                raise ValueError("Inconsistent data size")

    def _save_losses(self):
        """ For now just fitting format """ 
        
    def _transform_training_data(self):
        """ For now just fitting format """ 

class MultiAncestry(BaseMultiAncestry):
    def __init__(self, n_hidden1, n_hidden2, training_options={}):
        self.n_hidden1 = int(n_hidden1)
        self.n_hidden2 = int(n_hidden2)
        self.training_options = self._fill_training_options(training_options)
        self._initialize_save_losses()

    def fit(self, list_data, ancestry, seed=1993):
        """

        Parameters
        ----------
        list_data : list
            A n_regions length list of the SNP data for predicting ancestry.
            Each element should be a (n_individuals,n_snps) numpy array with
            a ***-1*** coding for a reference base while a 1 codes for the
            SNP.

        ancestry : np.array-like,shape=(n_individuals,n_races)
            The ancestry of each individual

        Returns
        -------
        self : MultiAncestryPT
            The object
        """
        self._test_inputs(list_data,ancestry)
        rng = np.random.default_rng(seed)
        ancestry = torch.tensor(ancestry)
        list_data_pt = [torch.tensor(data) for data in list_data]
        N, self.n_races = ancestry.shape
        self.n_regions = len(list_data)

        td = self.training_options
        n_epochs, n_iters = td["n_epochs"], td["n_iterations"]
        bs_reg, bs_samp = td["bs_region"], td["batch_size"]
        n_iters_i = td["n_inital_iterations"]
        n_iters_f = td["n_final_iterations"]

        self.ps = np.zeros(self.n_regions)

        shared_model = _SharedLayers(
            self.n_races, self.n_hidden1, self.n_hidden2
        )

        pred_loss = nn.CrossEntropyLoss()

        individual_models = []
        for i in range(self.n_regions):
            n_snvs = list_data[i].shape[1]
            self.ps[i] = n_snvs
            individual_layer = _IndividualLayers(n_snvs, self.n_hidden1)
            model_indiv = IndividualModel(shared_model, individual_layer)
            individual_models.append(model_indiv)

        for j in range(n_epochs):
            # This is the exterior loop for selecting regions of DNA
            region_select = rng.choice(self.n_regions, bs_reg, replace=False)

            training_variables = list(shared_model.parameters())
            training_variables_i = [] # type: ignore

            for region in region_select:
                training_variables = list(training_variables) + list(
                    individual_models[region].individual_layers.parameters()
                )
                training_variables_i = list(training_variables_i) + list(
                    individual_models[region].individual_layers.parameters()
                )

            optimizer = optim.Adam(training_variables, td["learning_rate"])
            optimizer_initial = optim.Adam(
                training_variables_i, td["learning_rate"]
            )

            list_data_sub = [list_data_pt[region] for region in region_select]
            network_sub = [
                individual_models[region] for region in region_select
            ]

            shared_model.eval()
            for i in range(n_iters_i):
                idx_select = rng.choice(N, bs_samp, replace=False)
                X_list_sub = [data[idx_select] for data in list_data_sub]
                Y_select = ancestry[idx_select]

                output = [network_sub[i](X_list_sub[i]) for i in range(bs_reg)]

                test_losses = [
                    pred_loss(output[i], Y_select) for i in range(bs_reg)
                ]

                loss = sum(test_losses)
                optimizer_initial.zero_grad()
                loss.backward()
                optimizer.step()

            shared_model.train()

            for i in trange(n_iters):
                idx_select = rng.choice(N, bs_samp, replace=False)
                X_list_sub = [data[idx_select] for data in list_data_sub]
                Y_select = ancestry[idx_select]

                output = [network_sub[i](X_list_sub[i]) for i in range(bs_reg)]

                test_losses = [
                    pred_loss(output[i], Y_select) for i in range(bs_reg)
                ]

                loss = sum(test_losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.shared_model = shared_model
        shared_model.eval()

        # Final training of first layers
        print("Final training of the region-specific layer")
        for i in trange(self.n_regions):
            X = list_data_pt[i]
            training_variables = list(
                individual_models[i].individual_layers.parameters()
            )
            optimizer = optim.Adam(training_variables, td["learning_rate"])

            for j in range(n_iters_f):
                idx_select = rng.choice(N, bs_samp, replace=False)
                X_batch = X[idx_select]
                Y_select = ancestry[idx_select]
                output = individual_models[i](X_batch)

                loss = pred_loss(output, Y_select)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            individual_models[i].eval()
        self.individual_models = individual_models

        self._store_instance_variables([shared_model,individual_models])
        return self

    def predict(self, list_data):
        """
        This predicts the local ancestry in every region of the genome

        Parameters
        ----------
        list_data : list
            A n_regions length list of the SNP data for predicting ancestry.
            Each element should be a (n_individuals,n_snps) numpy array with
            a ***-1*** coding for a reference base while a 1 codes for the
            SNP.

        Returns
        -------
        ancestries : np.array-like,(n_regions,)
            The ancestries of the different regions
        """
        log_probs = self.predict_logprob(list_data)
        if len(log_probs[0].shape) == 1:
            ancestry_hats = [
                np.argmax(log_probs[i]) for i in range(self.n_regions)
            ]
            ancestries = np.array(ancestry_hats)
        else:
            ancestry_hats = [
                np.argmax(log_probs[i], axis=1) for i in range(self.n_regions)
            ]
            ancestries = np.array(ancestry_hats)
        return ancestries

    def predict_individual_region(self, X_snv, idx):
        """
        This predicts the local ancestry in a specific region of the genome

        Parameters
        ----------
        X_snv : np.array-like,(n_spns[region],)
            A vector containing the snps for a specific region

        idx : int
            Region to predict the ancestry

        Returns
        -------
        ancestry : int
            The local ancestry
        """
        log_prob = self.predict_individual_region_logprob(X_snv, idx)
        ancestry = np.argmax(log_prob)
        return ancestry

    def predict_logprob(self, list_data):
        """
        This predicts the probability of local ancestry in every region of the
        genome

        Parameters
        ----------
        list_data : list
            A n_regions length list of the SNP data for predicting ancestry.
            Each element should be a (n_individuals,n_snps) numpy array with
            a ***-1*** coding for a reference base while a 1 codes for the
            SNP.

        Returns
        -------
        list_logprob : list
            A n_regions length list of ancestry log probabilities
        """
        list_logprob = [
            self.individual_models[i](list_data[i]).detach().numpy()
            for i in range(self.n_regions)
        ]
        return list_logprob

    def predict_individual_region_logprob(self, X_snv, idx):
        """
        This predicts the probability of local ancestry in a specific region
        of the genome

        Parameters
        ----------
        X_snv : np.array-like,(n_spns[region],)
            A vector containing the snps for a specific region

        idx : int
            Region to predict the ancestry

        Returns
        -------
        log_prob_ancestry : np.array-like,(n_ancestries,)
            The ancestry log probabilities
        """
        log_prob_ancestry = self.individual_models[idx](X_snv).detach().numpy()
        return log_prob_ancestry


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class _SharedLayers(nn.Module):
    """
    This is the set of layers shared between the different regions. Basically,
    there's a hidden latent representation that can be predicted from each
    set of SNVs and from that point ancestry is pretty much determined.
    """

    def __init__(self, n_ancestry, hidden1, hidden2):
        super().__init__()
        self.fc1 = nn.Linear(hidden1, hidden2)
        self.fc2 = nn.Linear(hidden2, hidden2)
        self.fc3 = nn.Linear(hidden2, n_ancestry)
        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.fc3)

    def forward(self, inputs):
        h = f.dropout(self.fc1(inputs), p=0.25, training=self.training)
        h = f.softplus(h)
        h = f.dropout(self.fc2(h), p=0.25, training=self.training)
        h = f.softplus(h)
        h = f.log_softmax(self.fc3(h))
        return h


class _IndividualLayers(nn.Module):
    """
    This is the individual layer of a particular region of the genome.
    Conceptually this just maps the specific set of SNVs into a common
    representation to sort through ancestry.
    """

    def __init__(self, n_2, hidden1):
        super().__init__()
        self.fc1 = nn.Linear(n_2, hidden1)
        init_weights(self.fc1)

    def forward(self, inputs):
        h = f.dropout(self.fc1(inputs), p=0.25, training=self.training)
        h = f.relu(h)
        return h


class IndividualModel(nn.Module):
    """
    This is the combined predictive model for a specific region of the genome,
    including both the shared and individual layers.
    """

    def __init__(self, shared_layers, individual_layers):
        super().__init__()
        self.individual_layers = individual_layers
        self.shared_layers = shared_layers

    def forward(self, inputs):
        h = self.individual_layers(inputs)
        h = self.shared_layers(h)
        return h
