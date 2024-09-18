import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize  # type: ignore
from typing import List, Union
from tqdm import trange
from scipy.stats import invwishart, multivariate_normal  # type: ignore


# Define the objective function to minimize: ||Ax - y||^2
def objective(x, A, y):
    return np.linalg.norm(A @ x - y) ** 2


# Define the constraint x_i >= c
def constraint(x, c):
    return x - c


def sample_from_data_list(
    data_list: List[np.ndarray], n_permutations: int
) -> List[np.ndarray]:
    """
    Randomly samples values from a list of data arrays, creating new
    sampled vectors.

    Parameters
    ----------
    data_list : List[np.ndarray]
        A list of numpy arrays, each representing a vector of data
        from which to sample.

    n_permutations : int
        The number of times to sample from each vector in `data_list`.

    Returns
    -------
    sampled_vectors : List[np.ndarray]
        A list of arrays, where each array is a sampled version of the
        original data vectors.
    """
    rng = np.random.default_rng()

    sampled_vectors: List[np.ndarray] = []

    for _ in range(n_permutations):
        sampled_vector = np.array([rng.choice(vector) for vector in data_list])
        sampled_vectors.append(sampled_vector)

    return sampled_vectors


class BatchAdaptationUnivariate:
    """
    The goal of this class is to perform domain adaptation to adjust for
    batch effects. Each batch contains one control sample, which is used
    to account for the mean shift in that batch. The method estimates
    parameters of the batch effects using variance components and adjusts
    the data accordingly.

    Parameters
    ----------
    ddof : int, default=-1
        Degrees of freedom correction for variance computation. Passed
        to `np.var`.

    n_permutations : int, default=1000
        The number of random permutations to be sampled from the input
        data to estimate variance components for the batch effects.
    """

    def __init__(
        self, ddof: int = -1, c: float = 0.5, n_permutations: int = 1000
    ) -> None:
        self.ddof = ddof
        self.n_permutations = n_permutations
        self.c = c

    def fit_transform(
        self, data_list: List[np.ndarray], controls: np.ndarray
    ) -> List[np.ndarray]:
        """
        Fits the batch effect model and applies the transformation to
        adjust for mean shifts.

        This method first fits the batch effect model by estimating the
        variance components of the controls and data. It then adjusts
        each data point by subtracting the mean shift induced by
        batch effects.

        Parameters
        ----------
        data_list : List[np.ndarray]
            A list of numpy arrays representing the data vectors from
            each batch.

        controls : np.ndarray
            A numpy array of control values corresponding to each batch.
            Used to estimate the mean and variance of batch effects.

        Returns
        -------
        data_adjusted : List[np.ndarray]
            A list of adjusted data vectors where the batch effect has
            been accounted for.
        """
        N_b = len(controls)
        if len(data_list) != N_b:
            raise ValueError("Dimension mismatch")

        controls_tilde = controls - np.mean(controls)
        self.mu_theta = np.mean(controls)

        sigma2_eps_theta = np.var(controls, ddof=self.ddof)
        random_sampled = sample_from_data_list(data_list, self.n_permutations)
        sigma2_eps_delta_theta = np.median(
            np.array(
                [np.var(random_sampled[i]) for i in range(len(random_sampled))]
            )
        )

        deviations = [data_list[i] - np.mean(data_list[i]) for i in range(N_b)]
        X_tilde = np.concatenate(deviations)
        sigma2_eps_delta = np.var(X_tilde, ddof=self.ddof)

        mat = np.zeros((3, 3))
        mat[:, 0] = 1
        mat[0] = 1
        mat[1, 1] = 1
        mat[2, 2] = 1
        mat[0, 1] = 1

        vec = np.zeros(3)
        vec[0] = sigma2_eps_delta_theta
        vec[1] = sigma2_eps_theta
        vec[2] = sigma2_eps_delta

        x0 = np.ones(mat.shape[1])
        cons = {"type": "ineq", "fun": lambda x: constraint(x, self.c)}
        result = minimize(
            objective, x0, args=(mat, vec), constraints=cons, method="SLSQP"
        )
        estimate = result.x

        self.sigma2_eps = estimate[0]
        self.sigma2_theta = estimate[1]
        self.sigma2_delta = estimate[2]

        self.sigma2_eps_delta_theta = sigma2_eps_delta_theta
        self.sigma2_eps_delta = sigma2_eps_delta
        self.sigma2_eps_theta = sigma2_eps_theta

        w = 1 - self.sigma2_eps / (self.sigma2_eps + self.sigma2_theta)
        if w > 1:
            w = 1
        elif w < 0:
            w = 0
        data_adjusted = [
            data_list[i] - w * controls_tilde[i] for i in range(N_b)
        ]
        self.w = w
        return data_adjusted

    def transform(
        self, data_new: np.ndarray, control: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Applies the batch effect transformation to new data using the
        previously fitted model.

        Parameters
        ----------
        data_new : np.ndarray
            A numpy array representing new data to be adjusted for batch
            effects.

        control : Union[float, np.ndarray]
            A control value or array representing the control for the new
            batch. Used to adjust for the mean shift in this batch.

        Returns
        -------
        data_adjusted : np.ndarray
            The new data adjusted for the batch effect using the
            previously fitted parameters.
        """
        data_adjusted = data_new - self.w * (control - self.mu_theta)
        return data_adjusted


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

    def fit(self, controls, X_list, theta_true=None,progress_bar=True):
        n_batches, p = controls.shape
        self.p = p
        if len(X_list) != n_batches:
            raise ValueError("Batch size mismatch")
        for i in range(n_batches):
            if X_list[i].shape[1] != p:
                raise ValueError("Batch %d has shape mismatch" % int(i))

        delta_samples = [X_list[i].copy() for i in range(n_batches)]
        theta_samples = np.zeros(controls.shape)

        self.Posterior_Sigma_delta_mean = np.zeros((self.p, self.p))
        self.Posterior_Sigma_epsilon_mean = np.zeros((self.p, self.p))
        self.Posterior_Sigma_theta_mean = np.zeros((self.p, self.p))
        delta_posterior_mean = [np.zeros(X_list[i].shape) for i in range(n_batches)]
        theta_posterior_mean = np.zeros(controls.shape)

        if progress_bar:
            print("Starting burnin sampling")
        for k in trange(self.n_burn, disable=not progress_bar):
            # Sample Sigma_delta
            Delta = np.vstack(delta_samples)
            S_delta = np.cov(Delta.T)
            nu_delta = p + self.nu_delta + Delta.shape[0]
            P_delta = self.Sigma_0_delta * np.eye(p) + S_delta * Delta.shape[0]
            print(S_delta)

            Sigma_delta = invwishart.rvs(df=nu_delta, scale=P_delta)

            # Sample Sigma_epsilon
            X_m_theta_delta = [
                X_list[j] - delta_samples[j] - theta_samples[j]
                for j in range(n_batches)
            ]
            Epsilon = np.vstack(X_m_theta_delta)
            S_epsilon = np.cov(Epsilon.T)
            nu_epsilon = p + self.nu_epsilon + Epsilon.shape[0]
            P_epsilon = (
                self.Sigma_0_epsilon * np.eye(p) + S_epsilon * Epsilon.shape[0]
            )
            Sigma_epsilon = invwishart.rvs(df=nu_delta, scale=P_epsilon)

            # Sample Sigma_theta
            S_theta = np.cov(theta_samples.T)
            #if theta_true is not None:
            #    S_theta = np.cov(theta_true.T)
            nu_theta = p + self.nu_theta + n_batches
            P_theta = self.Sigma_0_theta * np.eye(p) + S_theta * n_batches
            Sigma_theta = invwishart.rvs(df=nu_theta, scale=P_theta)

            # Sample theta_i
            Se_inv = la.inv(Sigma_epsilon)
            St_inv = la.inv(Sigma_theta)
            Sigma_posterior = la.inv(Se_inv + St_inv)
            Coef = np.dot(Sigma_posterior, Se_inv)
            for j in range(n_batches):
                mu_j = np.dot(Coef, controls[j])
                theta_samples[j] = multivariate_normal.rvs(
                    mean=mu_j, cov=Sigma_posterior
                )

            # Sample delta_ij
            Sd_inv = la.inv(Sigma_delta)
            Sigma_posterior = la.inv(Sd_inv + Se_inv)
            Coef = np.dot(Sigma_posterior, Se_inv)
            for j in range(n_batches):
                for k in range(X_list[j].shape[0]):
                    y_jk = X_list[j][k] - theta_samples[j]
                    mu_jk = np.dot(Coef, y_jk)
                    delta_samples[j][k] = multivariate_normal.rvs(
                        mean=mu_jk, cov=Sigma_posterior
                    )

        if progress_bar:
            print("Starting drawing samples")
        for k in trange(self.n_samples, disable=not progress_bar):
            if theta_true is not None:
                theta_samples = theta_true.copy()
            Delta = np.vstack(delta_samples)
            S_delta = np.cov(Delta.T)
            nu_delta = p + self.nu_delta + Delta.shape[0]
            P_delta = self.Sigma_0_delta * np.eye(p) + S_delta * Delta.shape[0]
            Sigma_delta = invwishart.rvs(df=nu_delta, scale=P_delta)

            # Sample Sigma_epsilon
            X_m_theta_delta = [
                X_list[j] - delta_samples[j] - theta_samples[j]
                for j in range(n_batches)
            ]
            Epsilon = np.vstack(X_m_theta_delta)
            S_epsilon = np.cov(Epsilon.T)
            nu_epsilon = p + self.nu_epsilon + Epsilon.shape[0]
            P_epsilon = (
                self.Sigma_0_epsilon * np.eye(p) + S_epsilon * Epsilon.shape[0]
            )
            Sigma_epsilon = invwishart.rvs(df=nu_delta, scale=P_epsilon)

            # Sample Sigma_theta
            S_theta = np.cov(theta_samples.T)
            if theta_true is not None:
                S_theta = np.cov(theta_true.T)
            nu_theta = p + self.nu_theta + n_batches
            P_theta = self.Sigma_0_theta * np.eye(p) + S_theta * n_batches
            Sigma_theta = invwishart.rvs(df=nu_theta, scale=P_theta)

            #Sigma_theta = np.eye(p)
            #Sigma_delta = np.eye(p)
            #Sigma_epsilon = np.eye(p)

            # Sample theta_i
            Se_inv = la.inv(Sigma_epsilon)
            St_inv = la.inv(Sigma_theta)
            Sigma_posterior = la.inv(Se_inv + St_inv)
            Coef = np.dot(Sigma_posterior, Se_inv)
            for j in range(n_batches):
                mu_j = np.dot(Coef, controls[j])
                theta_samples[j] = multivariate_normal.rvs(
                    mean=mu_j, cov=Sigma_posterior
                )

            # Sample delta_ij
            Sd_inv = la.inv(Sigma_delta)
            Sigma_posterior = la.inv(Sd_inv + Se_inv)
            Coef = np.dot(Sigma_posterior, Se_inv)
            for j in range(n_batches):
                for k in range(X_list[j].shape[0]):
                    y_jk = X_list[j][k] - theta_samples[j]
                    mu_jk = np.dot(Coef, y_jk)
                    delta_samples[j][k] = multivariate_normal.rvs(
                        mean=mu_jk, cov=Sigma_posterior
                    )
            
            self.Posterior_Sigma_delta_mean += Sigma_delta/self.n_samples
            self.Posterior_Sigma_epsilon_mean += Sigma_epsilon/self.n_samples
            self.Posterior_Sigma_theta_mean += Sigma_theta/self.n_samples
            theta_posterior_mean += theta_samples/self.n_samples
            for j in range(n_batches):
                delta_posterior_mean += delta_samples[j]/self.n_samples


        return delta_posterior_mean,theta_posterior_mean
