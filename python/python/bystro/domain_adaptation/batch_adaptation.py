import numpy as np
from scipy.optimize import minimize  # type: ignore
from typing import List, Union


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
