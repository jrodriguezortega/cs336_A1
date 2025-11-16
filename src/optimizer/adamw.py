import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr ({lr}) must be higher than 0")

        if betas[0] < 0 or betas[0] > 1 or betas[1] < 0 or betas[1] > 1:
            raise ValueError(
                f"Invalid value for momentum parameters ({betas})"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m, v = (
                    state.get("m", torch.zeros_like(grad)),
                    state.get("v", torch.zeros_like(grad)),
                )
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad**2)
                adjusted_lr = lr * (
                    math.sqrt(1 - pow(betas[1], t)) / (1 - pow(betas[0], t))
                )

                p.data -= adjusted_lr * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
