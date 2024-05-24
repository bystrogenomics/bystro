import pyro
import pyro.distributions as dist
from pyro.infer import mcmc, SVI, Trace_ELBO

import torch
from torch.optim import Adam

from pyro.distributions import Gamma, Normal  # type: ignore


class BasePrsCS:
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
        This fills in default MCMC options of the sampler. Further methods
        might override these but these are common/basic enough to leave in
        as an implemented method.

        Parameters
        ----------

        Returns
        -------

        """
        default_options = {
            "num_chains": 1,
            "num_warmup": 10,
            "num_samples": 50,
        }
        mopts = {**default_options, **mcmc_options}
        return mopts

    def _fill_hp_options(self, hp_options):
        default_options = {
            "a": 1.0,
            "b": 1.0,
        }
        hopts = {**default_options, **hp_options}
        return hopts


class PrsCSData(BasePrsCS):
    def fit(self, Z, y):
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



        nuts_kernel = mcmc.NUTS(model, step_size=5e-3,adapt_step_size=False)
        mcmc_run = mcmc.MCMC(nuts_kernel, num_samples=100, warmup_steps=20)
        mcmc_run.run(
            Z,
            y,
        )
        self.samples = mcmc_run.get_samples()
        return self

