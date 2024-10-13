import numpy as np

from bystro.domain_adaptation.batch_bayesian import BatchAdaptationBayesian


def simulate_data(n_batches=30, n_samples=15, p=3, seed=1993):
    rng = np.random.default_rng(seed)
    batch_effects = rng.multivariate_normal(
        mean=np.zeros(p), cov=3 * np.eye(p), size=n_batches
    )

    controls = batch_effects + rng.normal(size=(n_batches, p))
    true_vals = [rng.normal(size=(n_samples, p)) for i in range(n_batches)]
    X_list = [
        true_vals[i] + batch_effects[i] + rng.normal(size=(n_samples, p))
        for i in range(n_batches)
    ]
    return X_list, controls, true_vals, batch_effects


def test_bayesian():
    X_list, controls, true_vals, _batch_effects = simulate_data()

    model = BatchAdaptationBayesian(n_burn=2, n_samples=5)
    data_altered = model.fit_transform(X_list, controls)

    data_altered_stack = np.vstack(data_altered)
    data_original_stack = np.vstack(X_list)
    true_stack = np.vstack(true_vals)

    data_subtracted = [X_list[i] - controls[i] for i in range(len(X_list))]
    data_subtracted_stack = np.vstack(data_subtracted)

    error_adapted = np.mean((data_altered_stack - true_stack) ** 2)
    error_original = np.mean((data_original_stack - true_stack) ** 2)
    error_subtracted = np.mean((data_subtracted_stack - true_stack) ** 2)
    print(error_adapted)
    print(error_original)
    print(error_subtracted)
    assert error_adapted < error_original
    assert error_adapted < error_subtracted
