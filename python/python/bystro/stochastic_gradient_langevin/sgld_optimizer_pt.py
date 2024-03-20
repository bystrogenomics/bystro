import torch
from torch.optim.optimizer import Optimizer


class SGLDynamicsPT(Optimizer):
    """
    Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
    """

    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum + zero damping")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates
                the model and returns the loss.
        """
        if closure is not None:
            closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(momentum, buf) if nesterov else buf

                p.data.add_(d_p, alpha=-group["lr"])
                noise_std = torch.Tensor([2 * group["lr"]])
                noise_std = noise_std.sqrt()
                noise = (
                    p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std
                )
                p.data.add_(noise)

        return 1.0


class PreconditionedSGLDynamicsPT(Optimizer):
    """Implements pSGLD algorithm based on https://arxiv.org/pdf/1512.07666.pdf
    Built on the PyTorch RMSprop implementation
    (https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop)
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        beta=0.99,
        Lambda=1e-15,
        weight_decay=0,
        centered=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= Lambda:
            raise ValueError("Invalid epsilon value: {}".format(Lambda))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(
            lr=lr,
            beta=beta,
            Lambda=Lambda,
            centered=centered,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(PreconditionedSGLDynamicsPT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("centered", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is not None:
            closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "pSGLD does not support sparse gradients"
                    )
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["V"] = torch.zeros_like(p.data)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p.data)

                V = state["V"]
                beta = group["beta"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                V.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(beta).add_(1 - beta, grad)
                    G = (
                        V.addcmul(grad_avg, grad_avg, value=-1)
                        .sqrt_()
                        .add_(group["Lambda"])
                    )
                else:
                    G = V.sqrt().add_(group["Lambda"])

                p.data.addcdiv_(grad, G, value=-group["lr"])

                noise_std = 2 * group["lr"] / G
                noise_std = noise_std.sqrt()
                noise = (
                    p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std
                )
                p.data.add_(noise)

        return G


"""

MIT License

Copyright (c) 2020 SLIM group @ Georgia Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
