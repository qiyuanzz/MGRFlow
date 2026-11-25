"""Multilevel Morphology Aggregation (MMA)."""

import torch
from torch import nn


class MMA(nn.Module):
    """Project and fuse four aligned H&E feature grids along the channel axis."""

    def __init__(self, input_width: int, width: int, multilevel: bool):
        super().__init__()
        self.multilevel = multilevel
        self.branches = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(input_width), nn.Linear(input_width, width // 4))
            for _ in range(4)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(approximate="tanh"),
            nn.Linear(width, width),
        )
        self.position = nn.Parameter(torch.empty(1, 256, width))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # The last-layer control keeps the same projection layout.
        if not self.multilevel:
            features = features[:, -1:].expand(-1, 4, -1, -1)
        levels = [branch(features[:, i]) for i, branch in enumerate(self.branches)]
        return self.fusion(torch.cat(levels, dim=-1)) + self.position
