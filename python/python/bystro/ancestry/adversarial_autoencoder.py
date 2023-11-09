import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain

from tqdm import trange

Tensor = torch.FloatTensor


class Encoder(nn.Module):
    """
    This provides a deterministic function

    latent variables = encoder(data)

    Unlike a VAE, this is a deterministic rather than stochastic
    mapping. However, it is forced to approximate a distribution
    due to the adversarial loss with training
    """

    def __init__(self, observation_dimension, n_components):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(observation_dimension, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, n_components),
        )

    def forward(self, x):
        z = self.layers(x)
        return z


class Decoder(nn.Module):
    def __init__(self, observation_dimension, n_components):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_components, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, observation_dimension),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_components):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_components, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        predictions = self.model(z)
        return predictions


class AdversarialAutoencoder:
    """
    This implements an adversarial autoencoder
    """

    def __init__(self, n_components):
        self.n_components = int(n_components)

        self.encoder: Encoder | None = None
        self.decoder: Decoder | None = None

        self.training_options = {
            "learning_rate": 1e-3,
            "n_iterations": 1000,
            "b1": 0.5,
            "b2": 0.999,
            "batch_size": 200,
            "lambda": 0.01,
        }

    def fit(self, X, seed=2021):
        N, self.p = X.shape
        rng = np.random.default_rng(int(seed))
        X_ = torch.tensor(X, dtype=torch.float)
        lamb = self.training_options["lambda"]

        encoder = Encoder(self.p, self.n_components)
        decoder = Decoder(self.p, self.n_components)
        discriminator = Discriminator(self.n_components)  # Spelling corrected

        adversarial_loss = nn.BCELoss()
        generative_loss = nn.MSELoss()

        # Using chain to combine parameters from both models
        trainable_variables_g = chain(encoder.parameters(), decoder.parameters())
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
            Tensor(self.training_options["batch_size"], 1).fill_(1.0),
            requires_grad=False,
        )
        zeros = Variable(
            Tensor(self.training_options["batch_size"], 1).fill_(0.0),
            requires_grad=False,
        )

        self.losses_generative = np.zeros(self.training_options["n_iterations"])
        self.losses_discriminative = np.zeros(
            self.training_options["n_iterations"]
        )

        for i in trange(self.training_options["n_iterations"]):
            idx = rng.choice(N, size=self.training_options["batch_size"], replace=False)
            X_batch = X_[idx]
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

            real_z = Variable(torch.randn(self.training_options["batch_size"], self.n_components))

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
