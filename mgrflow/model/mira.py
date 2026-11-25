"""Morphology-Informed Relational Attention (MIRA) and the Flow block."""

import torch
from torch import nn


def modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tokens * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


def zero_linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MIRAInput(nn.Module):
    """Add projected H&E morphology at the matching input token positions."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.projection = zero_linear(width, width)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return tokens + self.projection(self.norm(context))[:, None]


class FlowBlock(nn.Module):
    """Spatial self-attention, H&E cross-attention, marker attention, and FFN."""

    def __init__(self, width: int, heads: int, mlp_ratio: float, mira: bool):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.spatial_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.context_norm = nn.LayerNorm(width, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.marker_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) if mira else None
        self.marker_attention = (
            nn.MultiheadAttention(width, heads, batch_first=True) if mira else None
        )
        hidden = int(width * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )
        branches = 4 if mira else 3
        self.modulation = nn.Sequential(nn.SiLU(), zero_linear(width, 3 * branches * width))

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, markers, positions, width = tokens.shape
        params = self.modulation(condition).chunk(12 if self.marker_attention else 9, dim=-1)
        # Flatten only the batch and marker axes for per-marker attention.
        flat = tokens.reshape(batch * markers, positions, width)
        shift, scale, gate = (part.reshape(batch * markers, width) for part in params[:3])
        query = modulate(self.spatial_norm(flat), shift, scale)
        spatial = self.spatial_attention(query, query, query, need_weights=False)[0]
        flat = flat + gate[:, None] * spatial

        # Protein tokens query the H&E grid throughout the transformer.
        shift, scale, gate = (part.reshape(batch * markers, width) for part in params[3:6])
        query = modulate(self.cross_norm(flat), shift, scale)
        he = context[:, None].expand(-1, markers, -1, -1).reshape(-1, positions, width)
        key_value = self.context_norm(he)
        cross = self.cross_attention(query, key_value, key_value, need_weights=False)[0]
        flat = flat + gate[:, None] * cross

        if self.marker_attention is not None:
            # Each spatial position attends over the requested marker set.
            tokens = flat.reshape(batch, markers, positions, width)
            relation = tokens.transpose(1, 2).reshape(batch * positions, markers, width)
            shift, scale, gate = (
                part[:, None].expand(-1, positions, -1, -1).reshape_as(relation)
                for part in params[6:9]
            )
            query = self.marker_norm(relation) * (1 + scale) + shift
            padding = None
            if marker_mask is not None:
                padding = (~marker_mask)[:, None].expand(-1, positions, -1)
                padding = padding.reshape(batch * positions, markers)
            related = self.marker_attention(
                query, query, query, key_padding_mask=padding, need_weights=False
            )[0]
            relation = relation + gate * related
            tokens = relation.reshape(batch, positions, markers, width).transpose(1, 2)
            flat = tokens.reshape(batch * markers, positions, width)

        shift, scale, gate = (part.reshape(batch * markers, width) for part in params[-3:])
        hidden = self.mlp(modulate(self.mlp_norm(flat), shift, scale))
        return (flat + gate[:, None] * hidden).reshape(batch, markers, positions, width)
