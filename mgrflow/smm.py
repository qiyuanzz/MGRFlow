"""Sequence-conditioned Marker Modulation (SMM)."""

import math

import torch
from torch import nn


class MarkerEmbedding(nn.Module):
    def __init__(self, table: torch.Tensor, width: int):
        super().__init__()
        self.register_buffer("table", table.detach().float().clone())
        self.network = nn.Sequential(
            nn.LayerNorm(table.shape[1]),
            nn.Linear(table.shape[1], width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, marker_ids: torch.Tensor) -> torch.Tensor:
        return self.network(self.table[marker_ids].to(self.network[1].weight.dtype))


class TimeEmbedding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, progress: torch.Tensor) -> torch.Tensor:
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(128, device=progress.device, dtype=torch.float32) / 127
        )
        angles = progress.float()[:, None] * frequencies * 1000
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.network(embedding.to(self.network[0].weight.dtype))


class SMM(nn.Module):
    """Combine one fixed sequence embedding per marker with the shared flow time."""

    def __init__(self, table: torch.Tensor, width: int):
        super().__init__()
        self.marker_embedding = MarkerEmbedding(table, width)
        self.time_embedding = TimeEmbedding(width)
        self.condition_fusion = nn.Sequential(nn.SiLU(), nn.Linear(width, width))

    def forward(self, marker_ids: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        marker = self.marker_embedding(marker_ids.flatten()).reshape(*marker_ids.shape, -1)
        time = self.time_embedding(progress)[:, None, :]
        return self.condition_fusion(marker + time)
