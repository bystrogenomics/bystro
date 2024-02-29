"""
This provides objects for scheduling the learning rate decay for stochastic
gradient descent

Objects
-------
None

Methods
-------
scheduler_sgld_geometric(lr=1e-2,lr_final=1e-4,n_samples=1e4)

"""


def scheduler_sgld_geometric(lr=2.5, lr_final=1e-1, n_samples=1e4):
    gamma = -0.55
    b = n_samples / ((lr_final / lr) ** (1 / gamma) - 1.0)
    a = lr / (b**gamma)

    def schedule(t, a=a, b=b, gamma=gamma):
        return a * ((b + t) ** gamma)

    return schedule
