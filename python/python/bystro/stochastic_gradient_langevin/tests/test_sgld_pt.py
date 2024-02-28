import numpy as np
from bystro.stochastic_gradient_langevin.sgld_optimizer_pt import (
    PreconditionedSGLDynamicsPT,
)
from bystro.stochastic_gradient_langevin.sgld_scheduler import (
    scheduler_sgld_geometric,
)
import torch
import copy

ptd = torch.distributions
device = torch.device("cpu")
torch.set_default_tensor_type("torch.FloatTensor")


def test_sgld():
    mu = torch.Tensor([1.2, 0.6], device=device)
    cov = (
        0.9
        * (torch.ones([2, 2], device=device) - torch.eye(2, device=device)).T
        + torch.eye(2, device=device) * 1.3
    )

    distn = ptd.MultivariateNormal(mu, covariance_matrix=cov)

    x = torch.zeros([2], requires_grad=True, device=device)
    n_samples = int(10000)

    lr_fn = scheduler_sgld_geometric(n_samples=n_samples)
    optimizer = PreconditionedSGLDynamicsPT([x], lr=1e-2, weight_decay=0.0)

    log_likelihood = np.zeros(n_samples)
    mean_samples = np.zeros((n_samples, 2))

    log_like = distn.log_prob
    counter = 0

    for i in range(n_samples):
        a = lr_fn(int(counter))
        for param_group in optimizer.param_groups:
            param_group["lr"] = a
        optimizer.zero_grad()
        loss = -1 * log_like(x)
        loss.backward()
        optimizer.step()
        log_likelihood[i] = loss.detach().numpy()
        mean_samples[i] = copy.deepcopy(x.data)
        counter += 1
