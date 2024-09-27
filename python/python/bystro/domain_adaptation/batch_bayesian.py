import numpy as np
import numpy.linalg as la
from scipy.stats import invwishart  # type: ignore
import torch
import pyro
from pyro.distributions import MultivariateNormal, LKJCholesky
from pyro.infer import MCMC, NUTS
from tqdm import trange


class BatchAdaptationBayesian:
    def __init__(
        self,
        nu_theta=2,
        nu_epsilon=2,
        nu_delta=2,
        Sigma_0_epsilon=1.0,
        Sigma_0_theta=1.0,
        Sigma_0_delta=1.0,
        n_samples=1000,
        n_burn=100,
    ):
        self.nu_theta = nu_theta
        self.nu_delta = nu_delta
        self.nu_epsilon = nu_epsilon
        self.Sigma_0_epsilon = Sigma_0_epsilon
        self.Sigma_0_theta = Sigma_0_theta
        self.Sigma_0_delta = Sigma_0_delta

        self.n_samples = int(n_samples)
        self.n_burn = int(n_burn)

        self.p = 0

    def fit_transform(self, X_list, controls):
        n_batches, p = controls.shape
        self.p = p
        if len(X_list) != n_batches:
            raise ValueError("Batch size mismatch")
        for i in range(n_batches):
            if X_list[i].shape[1] != p:
                raise ValueError("Batch %d has shape mismatch" % int(i))

        (
            Sigma_theta_list,
            Sigma_epsilon_list,
            Sigma_delta_list,
            theta_samples_list,
            w,
            w_list,
        ) = gibbs_sampler(
            X_list, controls, n_samp=self.n_samples, n_burn=self.n_burn
        )

        delta_estimates = compute_deltas(
            X_list, controls, Sigma_theta_list, Sigma_epsilon_list
        )

        self.Sigma_theta_list = Sigma_theta_list
        self.Sigma_epsilon_list = Sigma_epsilon_list
        self.Sigma_delta_list = Sigma_delta_list
        self.w = w
        self.w_list = w_list

        return delta_estimates

    def transform(self, X_list_new, controls_new):
        delta_estimates = compute_deltas(
            X_list_new,
            controls_new,
            self.Sigma_theta_list,
            self.Sigma_epsilon_list,
        )
        return delta_estimates


def sample_sigma_eps_sigma_delta(
    x_obs, y_obs, nu_eps, nu_delta, num_samples=1, warmup_steps=200
):
    """
    Function to return a single sample of Sigma_eps and Sigma_delta using MCMC in Pyro
    with Cholesky decomposition to avoid positive definite constraints.

    Parameters:
    - x_obs (torch.Tensor): Observed data for x (n_x, d).
    - y_obs (torch.Tensor): Observed data for y (n_y, d).
    - nu_eps (int): Degrees of freedom for the Wishart prior on Sigma_eps.
    - nu_delta (int): Degrees of freedom for the Wishart prior on Sigma_delta.
    - num_samples (int): Number of MCMC samples to draw (default: 1).
    - warmup_steps (int): Number of warm-up steps for MCMC (default: 200).

    Returns:
    - A single draw of Sigma_eps and Sigma_delta from the posterior.
    """

    def model(x_obs, y_obs, nu_eps, nu_delta):
        d = x_obs.shape[
            1
        ]  # Number of dimensions, should be the same for both x and y

        # Sample Cholesky factor for Sigma_eps (L_eps)
        L_eps = pyro.sample("L_eps", LKJCholesky(d, d + nu_eps))  # type: ignore

        # Sample Cholesky factor for Sigma_delta (L_delta)
        L_delta = pyro.sample("L_delta", LKJCholesky(d, d + nu_delta))  # type: ignore

        # Convert Cholesky factors to covariance matrices
        Sigma_eps = L_eps @ L_eps.T
        Sigma_delta = L_delta @ L_delta.T

        # Likelihood for x: p(x|Sigma_eps) = N(0, Sigma_eps)
        with pyro.plate("x_plate", x_obs.shape[0]):
            pyro.sample(
                "x",
                MultivariateNormal(  # type: ignore
                    torch.zeros_like(x_obs[0]), covariance_matrix=Sigma_eps
                ),
                obs=x_obs,
            )

        # Likelihood for y: p(y|Sigma_eps, Sigma_delta) = N(0, Sigma_eps + Sigma_delta)
        with pyro.plate("y_plate", y_obs.shape[0]):
            pyro.sample(
                "y",
                MultivariateNormal(  # type: ignore
                    torch.zeros_like(y_obs[0]),
                    covariance_matrix=Sigma_eps + Sigma_delta,
                ),
                obs=y_obs,
            )

    # Set up the NUTS kernel
    nuts_kernel = NUTS(model)

    # Run MCMC with the specified number of samples and warmup steps
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps,disable_progbar=True)
    mcmc.run(x_obs, y_obs, nu_eps, nu_delta)

    # Extract samples
    posterior_samples = mcmc.get_samples()

    # Return a single draw of Sigma_eps and Sigma_delta (the last sample)
    Sigma_eps_sample = (
        posterior_samples["L_eps"][-1] @ posterior_samples["L_eps"][-1].T
    )
    Sigma_delta_sample = (
        posterior_samples["L_delta"][-1] @ posterior_samples["L_delta"][-1].T
    )

    return Sigma_eps_sample.numpy(), Sigma_delta_sample.numpy()


def sample_sigma_theta(theta_samples, nu_theta):
    N, p = theta_samples.shape
    nu = N + p + nu_theta
    cov_emp = np.cov(theta_samples.T)
    Sig = N * cov_emp + np.eye(p)
    Sigma_theta = invwishart.rvs(df=nu, scale=Sig)
    return Sigma_theta


def sample_thetas(Sigma_theta, Sigma_epsilon, X, rng):
    Sigma_eps_inv = la.inv(Sigma_epsilon)
    Sigma_post = la.inv(la.inv(Sigma_theta) + Sigma_eps_inv)
    Coef_post = np.dot(Sigma_post, Sigma_eps_inv)
    Posterior_mean = np.dot(Coef_post, X.T).T
    samples_theta_i = np.zeros(X.shape)
    for i in range(X.shape[0]):
        samples_theta_i[i] = rng.multivariate_normal(
            mean=Posterior_mean[i], cov=Sigma_post
        )
    return samples_theta_i


def gibbs_sampler(Yl, cont, n_samp=100, n_burn=50):
    Sigma_theta_list = []
    Sigma_epsilon_list = []
    Sigma_delta_list = []
    theta_samples_list = []
    nb, p = cont.shape
    Sigma_theta = np.eye(p)
    Sigma_epsilon = np.eye(p)
    Sigma_delta = np.eye(p)
    theta_samples = 0.9 * cont.copy()
    Y_tilde = [Yl[i] - theta_samples[i] for i in range(nb)]
    w = np.zeros(p)
    w_list = []

    rng = np.random.default_rng(2021)

    for i in trange(n_burn):
        y_obs = torch.tensor(np.vstack(Y_tilde), dtype=torch.float)
        x_obs = torch.tensor(cont - theta_samples, dtype=torch.float)
        Sigma_epsilon, Sigma_delta = sample_sigma_eps_sigma_delta(
            x_obs, y_obs, 3, 3, num_samples=1, warmup_steps=20
        )
        Sigma_theta = sample_sigma_theta(theta_samples, 3)
        theta_samples = sample_thetas(Sigma_theta, Sigma_epsilon, cont, rng)
        Y_tilde = [Yl[i] - theta_samples[i] for i in range(nb)]

    for i in trange(n_samp):
        y_obs = torch.tensor(np.vstack(Y_tilde), dtype=torch.float)
        x_obs = torch.tensor(cont - theta_samples, dtype=torch.float)
        Sigma_epsilon, Sigma_delta = sample_sigma_eps_sigma_delta(
            x_obs, y_obs, 3, 3, num_samples=1, warmup_steps=20
        )
        Sigma_theta = sample_sigma_theta(theta_samples, 3)
        theta_samples = sample_thetas(Sigma_theta, Sigma_epsilon, cont, rng)
        Y_tilde = [Yl[i] - theta_samples[i] for i in range(nb)]

        Sigma_theta_list.append(Sigma_theta)
        Sigma_epsilon_list.append(Sigma_epsilon)
        Sigma_delta_list.append(Sigma_delta)
        theta_samples_list.append(theta_samples)

        Sig_eps = np.diag(Sigma_epsilon)
        Sig_the = np.diag(Sigma_theta)
        w_temp = Sig_the / (Sig_the + Sig_eps)
        w += w_temp / n_samp
        w_list.append(w_temp)

    return (
        Sigma_theta_list,
        Sigma_epsilon_list,
        Sigma_delta_list,
        theta_samples_list,
        w,
        w_list,
    )


def compute_deltas(X_list, controls, Sigma_theta_list, Sigma_epsilon_list):
    delta_ests = [np.zeros(X_list[i].shape) for i in range(len(X_list))]
    n_samples = len(Sigma_epsilon_list)
    n_batches = len(X_list)
    for i in range(n_samples):
        Sig_eps = np.diag(Sigma_epsilon_list[i])
        Sig_the = np.diag(Sigma_theta_list[i])
        w = Sig_the / (Sig_the + Sig_eps)
        for j in range(n_batches):
            delta_estimate = X_list[j] - w * controls[j]
            delta_ests[j] += delta_estimate / n_samples

    return delta_ests
