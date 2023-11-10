import numpy as np
import scipy.stats as st
from bystro.ancestry.adversarial_autoencoder import AdversarialAutoencoder

def generate_data():
    rng = np.random.default_rng(2021)
    N = 10000
    p = 100
    L = 3
    sigma = 1.0
    W_base = st.ortho_group.rvs(p)
    W = W_base[:L]
    lamb = rng.gamma(1, 1, size=L) + 1
    for ll in range(L):
        W[ll] = W[ll] * lamb[ll]
    S_train = rng.normal(size=(N, L))
    X_hat = np.dot(S_train, W)
    X_noise = sigma * rng.normal(size=(N, p))
    X_train = X_hat + X_noise
    return W, sigma, X_train, S_train

def test_adversarial_autoencoder():
    W,sigma, X,S_train = generate_data()
    model = AdversarialAutoencoder(3)
    model.fit(X)

    S_est = model.transform(X.astype(np.float32))
    plt.scatter(S_est[:,0],S_est[:,1])

	
if __name__ == "__main__":
    test_adversarial_autoencoder()
