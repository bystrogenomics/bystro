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
            "num_warmup": 1000,
            "num_samples": 5000,
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

            beta_variance = sigma2 / (N * psi)
            beta = pyro.sample(
                "beta",
                Normal(0.0, beta_variance.sqrt()).expand([p]).to_event(1),
            )

            with pyro.plate("data", N):
                mean = torch.matmul(Z, beta)
                pyro.sample("obs", Normal(mean, sigma2.sqrt()), obs=y)

        nuts_kernel = mcmc.NUTS(model)
        mcmc_run = mcmc.MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
        mcmc_run.run(
            y,
            Z,
        )
        self.samples = mcmc_run.get_samples()
        return self


class PrsCSDataVariational(BasePrsCS):
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

            beta_variance = sigma2 / (N * psi)
            beta = pyro.sample(
                "beta",
                Normal(0.0, beta_variance.sqrt()).expand([p]).to_event(1),
            )

            with pyro.plate("data", N):
                mean = torch.matmul(Z, beta)
                pyro.sample("obs", Normal(mean, sigma2.sqrt()), obs=y)

        def guide(Z, y):  # noqa: ARG001
            w_loc = pyro.param(
                "w_loc", torch.tensor(1.0), constraint=dist.constraints.positive
            )
            w_scale = pyro.param(
                "w_scale",
                torch.tensor(0.1),
                constraint=dist.constraints.positive,
            )
            phi_loc = pyro.param(
                "phi_loc",
                torch.tensor(1.0),
                constraint=dist.constraints.positive,
            )
            phi_scale = pyro.param(
                "phi_scale",
                torch.tensor(0.1),
                constraint=dist.constraints.positive,
            )
            sigma2_loc = pyro.param(
                "sigma2_loc",
                torch.tensor(1.0),
                constraint=dist.constraints.positive,
            )
            sigma2_scale = pyro.param(
                "sigma2_scale",
                torch.tensor(0.1),
                constraint=dist.constraints.positive,
            )

            delta_loc = pyro.param(
                "delta_loc", torch.ones(p), constraint=dist.constraints.positive
            )
            delta_scale = pyro.param(
                "delta_scale",
                torch.ones(p) * 0.1,
                constraint=dist.constraints.positive,
            )
            psi_loc = pyro.param(
                "psi_loc", torch.ones(p), constraint=dist.constraints.positive
            )
            psi_scale = pyro.param(
                "psi_scale",
                torch.ones(p) * 0.1,
                constraint=dist.constraints.positive,
            )

            w = pyro.sample("w", Gamma(w_loc, w_scale))  # noqa: F841
            phi = pyro.sample("phi", Gamma(phi_loc, phi_scale))  # noqa: F841
            sigma2 = pyro.sample("sigma2", Gamma(sigma2_loc, sigma2_scale))
            delta = pyro.sample(  # noqa: F841
                "delta",
                Gamma(delta_loc, delta_scale).expand([p]).to_event(1),
            )
            psi = pyro.sample(
                "psi", Gamma(psi_loc, psi_scale).expand([p]).to_event(1)
            )

            beta_loc = pyro.param("beta_loc", torch.zeros(p))
            beta_scale = pyro.param(
                "beta_scale",
                torch.ones(p),
                constraint=dist.constraints.positive,
            )
            beta_variance = sigma2 / (N * psi)  # noqa: F841
            beta = pyro.sample(  # noqa: F841
                "beta",
                Normal(beta_loc, beta_scale).expand([p]).to_event(1),
            )

        pyro.clear_param_store()
        svi = SVI(model, guide, Adam([{"lr": 0.01}]), loss=Trace_ELBO())

        num_iterations = 1000
        for j in range(num_iterations):
            loss = svi.step(Z, y)
            if j % 100 == 0:
                print(f"Iteration {j}: loss = {loss}")

        return self
