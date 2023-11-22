"""
Background
----------
Adversarial autoencoders (AAEs) represent a novel approach to 
unsupervised learning by combining the principles of autoencoders 
with adversarial networks. Traditional autoencoders aim to compress 
input data into a lower-dimensional latent space and then reconstruct 
the original input from this compressed representation. AAEs introduce 
an adversarial training component by incorporating a generative 
adversarial network (GAN) into the autoencoder architecture. The 
adversarial network's role is to discriminate between the encoded 
latent representations and samples drawn from a predefined distribution. 
This adversarial process encourages the autoencoder to produce latent 
representations that closely resemble samples from the specified 
distribution, thereby promoting the learning of a more structured and 
meaningful latent space.

The adversarial component in AAEs helps overcome some limitations of 
standard autoencoders, such as mode collapse and lack of diversity in 
the generated samples. By introducing adversarial training, AAEs can 
learn a more robust and continuous latent space that captures the 
underlying structure of the input data. This combination of autoencoder 
and GAN principles makes adversarial autoencoders a powerful tool for 
tasks like data generation, anomaly detection, and representation 
learning, where learning a meaningful and compact latent representation 
is crucial for effective performance.

This implements an adversarial encoder and the objects

Objects
-------
Encoder(nn.Module)
    This provides a deterministic function
    latent variables = encoder(data)

Decoder(nn.Module)
    This provides a deterministic function
    reconstructed data = decoder(latent_variables)

Discriminator(nn.Module)
    This defines a neural network that distinguishes
    latent variables computed from our encoder on observed
    data and data from a synthetic distribution


AdversarialAutoencoder
    This fits an adversarial autoencoder given data


Methods
-------
None
"""
from typing import Any
import numpy as np
from numpy.typing import NDArray

import torch
from torch import tensor
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain

from tqdm import trange

from sklearn.mixture import GaussianMixture

Tensor = torch.FloatTensor


class Encoder(nn.Module):
    """
    This provides a deterministic function

    latent variables = encoder(data)

    Unlike a VAE, this is a deterministic rather than stochastic
    mapping. However, it is forced to approximate a distribution
    due to the adversarial loss with training
    """

    def __init__(self, observation_dimension, n_components, encoder_options):
        super().__init__()
        eo = encoder_options

        self.layers = nn.Sequential(
            nn.Linear(observation_dimension, eo["n_nodes"]),
            nn.LeakyReLU(0.2),
            nn.Linear(eo["n_nodes"], eo["n_nodes"]),
            nn.BatchNorm1d(eo["n_nodes"]),
            nn.LeakyReLU(0.2),
            nn.Linear(eo["n_nodes"], n_components),
        )

    def forward(self, x):
        z = self.layers(x)
        return z


class Decoder(nn.Module):
    """
    This provides a deterministic function

    reconstructed data = decoder(latent_variables)
    """

    def __init__(self, observation_dimension, n_components, decoder_options):
        super().__init__()
        do = decoder_options

        self.model = nn.Sequential(
            nn.Linear(n_components, do["n_nodes"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(do["n_nodes"], do["n_nodes"]),
            nn.BatchNorm1d(do["n_nodes"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(do["n_nodes"], observation_dimension),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(nn.Module):
    """
    This defines a neural network that distinguishes
    latent variables computed from our encoder on observed
    data and data from a synthetic distribution
    """

    def __init__(self, n_components, discriminator_options):
        super().__init__()
        do = discriminator_options

        self.model = nn.Sequential(
            nn.Linear(n_components, do["n_nodes"]),
            nn.LeakyReLU(0.2),
            nn.Linear(do["n_nodes"], do["n_nodes2"]),
            nn.LeakyReLU(0.2),
            nn.Linear(do["n_nodes2"], 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        predictions = self.model(z)
        return predictions


class AdversarialAutoencoder:
    """
    This implements an adversarial autoencoder
    """

    def __init__(
        self,
        n_components,
        training_options: dict[str, Any] | None = None,
        latent_distribution_options: dict[str, Any] | None = None,
        encoder_options: dict[str, Any] | None = None,
        decoder_options: dict[str, Any] | None = None,
        discriminator_options: dict[str, Any] | None = None,
    ):
        self.n_components = int(n_components)

        self.encoder: Encoder | None = None
        self.decoder: Decoder | None = None

        if training_options is None:
            training_options = {}
        if encoder_options is None:
            encoder_options = {}
        if decoder_options is None:
            decoder_options = {}
        if discriminator_options is None:
            discriminator_options = {}
        if latent_distribution_options is None:
            latent_distribution_options = {}

        self._fill_training_options(training_options)
        self._fill_model_options(
            latent_distribution_options,
            encoder_options,
            decoder_options,
            discriminator_options,
        )

    def fit(self, X, seed=2021):
        N, self.p = X.shape
        rng = np.random.default_rng(int(seed))
        X_ = tensor(X, dtype=torch.float)
        lamb = self.training_options["lambda"]

        n_iterations = int(self.training_options["n_iterations"])
        batch_size = int(self.training_options["batch_size"])

        encoder = Encoder(self.p, self.n_components, self.encoder_options)
        decoder = Decoder(self.p, self.n_components, self.decoder_options)
        discriminator = Discriminator(
            self.n_components, self.discriminator_options
        )

        adversarial_loss = nn.BCELoss()
        generative_loss = nn.MSELoss()

        # Using chain to combine parameters from both models
        trainable_variables_g = chain(
            encoder.parameters(), decoder.parameters()
        )
        trainable_variables_d = discriminator.parameters()

        optimizer_G = torch.optim.Adam(
            trainable_variables_g,
            lr=self.training_options["learning_rate"],
            betas=(self.training_options["b1"], self.training_options["b2"]),
        )
        optimizer_D = torch.optim.Adam(
            trainable_variables_d,
            lr=self.training_options["learning_rate"],
            betas=(self.training_options["b1"], self.training_options["b2"]),
        )

        ones = Variable(
            Tensor(batch_size, 1).fill_(1.0),
            requires_grad=False,
        )
        zeros = Variable(
            Tensor(batch_size, 1).fill_(0.0),
            requires_grad=False,
        )

        self.losses_generative = np.zeros(n_iterations)
        self.losses_discriminative = np.zeros(n_iterations)

        XX = rng.normal(scale=.3,size=(10000,self.n_components))
        XX[:3300,0] += 5
        XX[3300:6600,0] += -5
        gmm = GaussianMixture(3)
        gmm.fit(XX)

        for i in trange(n_iterations):
            idx = rng.choice(
                N, size=batch_size, replace=False
            )
            X_batch = X_[tensor(idx)]
            Z = encoder(X_batch)
            X_recon = decoder(Z)

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            prediction_real_data = discriminator(Z)
            gloss = generative_loss(X_recon, X_batch)
            dloss = adversarial_loss(prediction_real_data, ones)
            G_loss = lamb * gloss + (1 - lamb) * dloss
            G_loss.backward()
            optimizer_G.step()

            samples,_ = gmm.sample(n_samples=batch_size)
            real_z = Variable(tensor(samples.astype(np.float32)))

            real_loss = adversarial_loss(discriminator(real_z), ones)
            fake_loss = adversarial_loss(discriminator(Z.detach()), zeros)

            D_loss = 0.5 * (real_loss + fake_loss)

            D_loss.backward()
            optimizer_D.step()

            self.losses_generative[i] = gloss.item()
            self.losses_discriminative[i] = dloss.item()

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        return self

    def transform(self, X: NDArray) -> NDArray[np.float_]:
        """
        This returns the latent variable estimates given X

        Parameters
        ----------
        X : NDArray,(N_samples,p)
            The data to transform.

        Returns
        -------
        S : NDArray,(N_samples,n_components)
            The factor estimates
        """
        if self.encoder is None:
            raise ValueError("The model has not been fit yet")

        X_ = tensor(X)
        S_ = self.encoder(X_)
        S = S_.detach().numpy()
        return S

    def inverse_transform(self, S: NDArray) -> NDArray[np.float_]:
        """
        This returns the reconstruction given latent variables

        Parameters
        ----------
        S : NDArray,(N_samples,n_components)
            The factor estimates

        Returns
        -------
        X_recon : np array-like,(N_samples,p)
            The reconstruction
        """
        if self.decoder is None:
            raise ValueError("The model has not been fit yet")

        S_ = tensor(S)
        X_ = self.decoder(S_)
        X_recon = X_.detach().numpy()

        return X_recon

    def _fill_training_options(self, training_options: dict[str, Any]) -> None:
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

        batch_size : int, default=None
            The number of observations to use at each iteration. If none
            corresponds to batch learning
        """
        default_options = {
            "n_iterations": 3000,
            "learning_rate": 1e-2,
            "batch_size": 100,
            "b1": 0.5,
            "b2": 0.999,
            "lambda": 0.1,
        }
        tops = {**default_options, **training_options}

        default_keys = set(default_options.keys())
        final_keys = set(tops.keys())

        expected_but_missing_keys = default_keys - final_keys
        unexpected_but_present_keys = final_keys - default_keys
        if expected_but_missing_keys:
            raise ValueError(
                "the following training options were expected but not found..."
            )
        if unexpected_but_present_keys:
            raise ValueError(
                "the following training options were unrecognized but provided..."
            )

        self.training_options = tops

    def _fill_model_options(
        self,
        latent_distribution_options: dict[str, Any],
        encoder_options: dict[str, Any],
        decoder_options: dict[str, Any],
        discriminator_options: dict[str, Any],
    ) -> None:
        """
        This sets the default parameters for our encoder, decoder and discriminator

        Parameters
        ----------
        latent_distribution_options: dict[str, Any]
            Latent distribution options

        encoder_options: dict[str, Any]
            Encoder parameters

        decoder_options: dict[str, Any]
            Decoder parameters

        discriminator_options: dict[str, Any]
            Discriminator parameters
        """
        default_latent_distribution_options = {
            "n_iterations": 3000,
        }
        default_encoder_options = {
            "n_nodes": 128,
        }
        default_decoder_options = {
            "n_nodes": 128,
        }
        default_discriminator_options = {
            "n_nodes": 64,
            "n_nodes2": 16,
        }
        self.latent_distribution_options = {
            **default_latent_distribution_options,
            **latent_distribution_options,
        }
        self.encoder_options = {**default_encoder_options, **encoder_options}
        self.decoder_options = {**default_decoder_options, **decoder_options}
        self.discriminator_options = {
            **default_discriminator_options,
            **discriminator_options,
        }
