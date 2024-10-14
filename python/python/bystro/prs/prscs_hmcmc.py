import pyro
import pyro.distributions as dist
from pyro.infer import mcmc

import torch

from pyro.distributions import Gamma, Normal  # type: ignore


class BasePrsCS:
    """
    This implements the base class for PRS-CS methods. Currently, this just
    sets basic MCMC and parameter values along with defining attributes
    to be set during training. Future work will more closely integrate
    this with the template objects used throughout bystro.
    """

    def __init__(
        self,
        mcmc_options=None,
        hp_options=None,
    ):
        if mcmc_options is None:
            mcmc_options = {}
        if hp_options is None:
            hp_options = {}
        self.mcmc_options = self._fill_mcmc_options(mcmc_options)
        self.hp_options = self._fill_hp_options(hp_options)
        self.samples = None
        self._model = None
        self.model_kwargs = None

    def _fill_mcmc_options(self, mcmc_options):
        """
        This fills in the MCMC parameters

        Parameters
        ----------
        num_chains : int
            Number of chains to run

        num_warmup : int
            Number of burnin samples

        num_samples : int
            Number of MCMC samples to keep

        Returns
        -------
        mopts : dict
            The dictionary of completed options
        """
        default_options = {
            "num_chains": 1,
            "warmup_steps": 1000,
            "num_samples": 5000,
        }
        mopts = {**default_options, **mcmc_options}
        return mopts

    def _fill_hp_options(self, hp_options):
        """
        This fills in the hyperparameters

        Parameters
        ----------
        a : float
            Shape of the gamma distribution

        b : float
            Scale of the gamma distribution

        Returns
        -------
        hopts : dict
            The dictionary of completed options
        """
        default_options = {
            "a": 1.0,
            "b": 1.0,
        }
        hopts = {**default_options, **hp_options}
        return hopts


class PrsCSData(BasePrsCS):
    """
    This implements PRS-CS given the original samples as opposed to summary
    statistics. This allows for easier potential future customization on
    the predictive loss (for example allowing for more robust alternatives
    to an L2 loss or classification.
    """

    def fit(self, Z, y):
        """
        The fit method.

        Parameters
        ----------
        Z : np.array-like,(n_samples,p)
            The covariates of interest

        y : np.array-like,(n_samples,)
            The trait, disease, or outcome

        Returns
        -------
        self
        """
        hp = self.hp_options
        N, p = Z.shape

        Z = torch.tensor(Z)
        y = torch.tensor(y)

        def model(Z, y):
            w = pyro.sample("w", Gamma(0.5, 1.0))
            phi = pyro.sample("phi", Gamma(0.5, w))
            sigma2 = pyro.sample("sigma2", dist.InverseGamma(0.5, 0.5))

            delta = pyro.sample(
                "delta", Gamma(hp["b"], phi).expand([p]).to_event(1)
            )
            psi = pyro.sample(
                "psi", Gamma(hp["a"], delta).expand([p]).to_event(1)
            )

            beta_variance = sigma2 * psi
            beta = pyro.sample("beta", Normal(0, beta_variance).to_event(1))

            beta = beta.reshape(-1, 1)

            with pyro.plate("data", N):
                prediction_mean = torch.matmul(Z, beta).squeeze()
                pyro.sample("obs", Normal(prediction_mean, sigma2), obs=y)

        nuts_kernel = mcmc.NUTS(model, step_size=5e-3, adapt_step_size=False)
        mcmc_run = mcmc.MCMC(nuts_kernel, **self.mcmc_options)
        mcmc_run.run(
            Z,
            y,
        )
        self.samples = mcmc_run.get_samples()
        return self
