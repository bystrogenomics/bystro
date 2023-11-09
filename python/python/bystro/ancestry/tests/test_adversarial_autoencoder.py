import numpy as np

from bystro.ancestry.adversarial_autoencoder import AdversarialAutoencoder


def test_adversarial_autoencoder_convergence():
    # Parameters
    n_samples = 1000
    observation_dimension = 10
    n_components = 3
    n_iterations = 100
    batch_size = 50

    # Generate synthetic data (simple Gaussian distribution for testing)
    rng = np.random.default_rng()  # Create a random number generator instance
    X_synthetic = rng.standard_normal((n_samples, observation_dimension)).astype(np.float32)

    # Instantiate the autoencoder
    AAE = AdversarialAutoencoder(n_components)
    AAE.training_options["n_iterations"] = n_iterations
    AAE.training_options["batch_size"] = batch_size
    AAE.training_options["learning_rate"] = 0.001

    # Train the autoencoder
    AAE.fit(X_synthetic)

    # Assert that the generative and discriminative losses have decreased
    assert AAE.losses_generative[-1] < AAE.losses_generative[0], "Generative loss did not decrease"
    assert (
        AAE.losses_discriminative[-1] < AAE.losses_discriminative[0]
    ), "Discriminative loss did not decrease"

    assert AAE.encoder is not None, "Encoder was not initialized"
    assert AAE.decoder is not None, "Decoder was not initialized"


# Run the test
test_adversarial_autoencoder_convergence()
