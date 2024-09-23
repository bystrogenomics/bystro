import numpy as np

from bystro.domain_adaptation.batch_adaptation import BatchAdaptationUnivariate


def simulate_data(
    n_batches=30,
    n_samples=15,
    seed=2021,
    sigma_d=1.0,
    sigma_theta=1.0,
    sigma_epsilon=1.0,
    sigma_delta=1.0,
):
    rng = np.random.default_rng(seed)
    theta_list = sigma_theta * rng.normal(size=n_batches)
    control_list = sigma_epsilon * rng.normal(size=n_batches) + theta_list
    data_list = []
    delta_list = []
    for i in range(n_batches):
        delta = sigma_delta * rng.normal(size=n_samples)
        delta_list.append(delta)
        data = delta + sigma_d * rng.normal(size=n_samples) + theta_list[i]
        data_list.append(data)

    return theta_list, control_list, data_list, delta_list


def test_univariate_adaptation():
    sigma_epsilon = 1.0
    sigma_theta = 1.8
    sigma_delta = 1.0
    seed = 1993

    theta_list, control_list, data_list, delta_list = simulate_data(
        n_batches=70,
        n_samples=15,
        seed=seed,
        sigma_theta=sigma_theta,
        sigma_epsilon=sigma_epsilon,
        sigma_delta=sigma_delta,
    )
    model = BatchAdaptationUnivariate()
    data_altered = model.fit_transform(data_list, control_list)

    data_altered_stack = np.concatenate(data_altered)
    data_original_stack = np.concatenate(data_list)
    delta_stack = np.concatenate(delta_list)
    data_subtracted = np.concatenate(
        [data_list[i] - control_list[i] for i in range(len(data_list))]
    )
    error_adapted = np.mean((data_altered_stack - delta_stack) ** 2)
    error_original = np.mean((data_original_stack - delta_stack) ** 2)
    error_subtracted = np.mean((data_subtracted - delta_stack) ** 2)
    assert error_adapted < error_original
    assert error_adapted < error_subtracted

    data_new = model.transform(data_list[0], control_list[0])
    adapted_error = np.mean((data_new - delta_list[0]) ** 2)
    original_error = np.mean((data_list[0] - delta_list[0]) ** 2)

    assert adapted_error < original_error
