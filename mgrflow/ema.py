"""Optional exponential moving average of the trainable Flow weights."""

from copy import deepcopy

import torch


class EMA:
    def __init__(self, model, decay: float = 0.9999):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between 0 and 1.")
        self.decay = decay
        self.model = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        torch._foreach_lerp_(
            list(self.model.parameters()), list(model.parameters()), 1 - self.decay
        )
