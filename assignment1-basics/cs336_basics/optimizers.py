from torch.optim import Optimizer
import torch
from collections.abc import Callable

class AdamW(Optimizer):
    def __init__(self, params, lr: float=0.001, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p)) # first order momentum
                v = state.get("v", torch.zeros_like(p)) # second order momentum
                t = state.get("t", 1)
                grad = p.grad
                beta1, beta2 = betas
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                p.data -= lr_t * (m / (v ** 0.5 + eps))
                p.data -= lr * weight_decay * p.data
                t += 1
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss
            
            