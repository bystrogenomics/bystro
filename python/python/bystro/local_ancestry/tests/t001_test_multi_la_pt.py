import numpy as np
from bystro.local_ancestry.multi_ancestry import MultiAncestry

import torch
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.datasets import fetch_openml # type: ignore


def one_hot(arr):
    vals = np.unique(arr)
    n_vals = len(vals)
    N = len(arr)
    out = np.zeros((N, n_vals))
    for i in range(n_vals):
        out[arr == vals[i], i] = 1
    return out

rng = np.random.default_rng(2021)

X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
)
X = np.round(X / 255)
X[X == 0] = -1

y_oh = one_hot(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_oh, train_size=60000, test_size=10000
)
n_regions = 200
idx_list = [rng.choice(784, size=50, replace=False) for i in range(n_regions)]
list_data = [
    X_train[:, idx_list[i]].astype(np.float32) for i in range(n_regions)
]
list_data_test = [
    torch.tensor(X_test[:, idx_list[i]].astype(np.float32))
    for i in range(n_regions)
]

training_options = {
    "learning_rate": 1e-3,
    "n_iterations": 5,
    "n_inital_iterations": 3,
    "n_final_iterations": 3,
    "n_epochs": 3,
    "batch_size": 100,
    "bs_region": 10,
}

model = MultiAncestry(100, 50, training_options=training_options)

model.fit(list_data, torch.tensor(y_train))
y_hats = model.predict(list_data_test)
y_tests = np.zeros(y_test.shape[0])
for i in range(10):
    y_tests[y_test[:, i] == 1] = i
