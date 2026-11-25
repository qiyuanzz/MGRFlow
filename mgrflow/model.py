"""Joint rectified flow over a variable panel of protein latents."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .mira import FlowBlock, MIRAInput, modulate, zero_linear
from .mma import MMA
from .smm import SMM

LATENT_SHAPE = (4, 32, 32)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    pathology_width: int = 1536
    smm: bool = True
    mma: bool = True
    mira: bool = True
    activation_checkpointing: bool = False


class MGRFlow(nn.Module):
    """Predict jointly a four-channel velocity field for each requested marker."""

    def __init__(self, marker_table: torch.Tensor, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        width = config.width
        if width % 4:
            raise ValueError("The Flow width must be divisible by the four MMA levels.")
        table = marker_table if config.smm else torch.eye(len(marker_table))
        self.mma = MMA(config.pathology_width, width, config.mma)
        self.smm = SMM(table, width)
        self.input_projection = nn.Linear(16, width)
        self.position = nn.Parameter(torch.empty(1, 1, 256, width))
        nn.init.normal_(self.position, std=0.02)
        self.mira_input = MIRAInput(width) if config.mira else None
        self.blocks = nn.ModuleList(
            FlowBlock(width, config.heads, config.mlp_ratio, config.mira)
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), zero_linear(width, 2 * width))
        self.output_projection = zero_linear(width, 16)

    @staticmethod
    def patchify(latents: torch.Tensor) -> torch.Tensor:
        """Rearrange [B, K, 4, 32, 32] into [B, K, 256, 16]."""
        batch, markers = latents.shape[:2]
        patches = latents.reshape(batch, markers, 4, 16, 2, 16, 2)
        return patches.permute(0, 1, 3, 5, 2, 4, 6).reshape(batch, markers, 256, 16)

    @staticmethod
    def unpatchify(tokens: torch.Tensor) -> torch.Tensor:
        batch, markers = tokens.shape[:2]
        grid = tokens.reshape(batch, markers, 16, 16, 4, 2, 2)
        return grid.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch, markers, *LATENT_SHAPE)

    def forward(
        self,
        state: torch.Tensor,
        progress: torch.Tensor,
        he_features: torch.Tensor,
        marker_ids: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.velocity(state, progress, self.mma(he_features), marker_ids, marker_mask)

    def velocity(
        self,
        state: torch.Tensor,
        progress: torch.Tensor,
        context: torch.Tensor,
        marker_ids: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.smm(marker_ids, progress)
        tokens = self.input_projection(self.patchify(state)) + self.position
        if self.mira_input is not None:
            tokens = self.mira_input(tokens, context)
        for block in self.blocks:
            inputs = (tokens, context, condition, marker_mask)
            if self.config.activation_checkpointing and self.training:
                tokens = checkpoint(block, *inputs, use_reentrant=False)
            else:
                tokens = block(*inputs)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        width = self.config.width
        flat = tokens.reshape(-1, 256, width)
        adapted = modulate(
            self.final_norm(flat), shift.reshape(-1, width), scale.reshape(-1, width)
        )
        velocity = self.output_projection(adapted)
        return self.unpatchify(velocity.reshape(*state.shape[:2], 256, 16))

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        he_features: torch.Tensor,
        marker_ids: torch.Tensor,
        *,
        steps: int = 4,
        marker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler integration from Gaussian noise (progress 0) to proteins (1)."""
        if steps < 1:
            raise ValueError("Use at least one sampling step.")
        context = self.mma(he_features)
        state = noise
        for step in range(steps):
            progress = state.new_full((state.shape[0],), step / steps, dtype=torch.float32)
            velocity = self.velocity(state, progress, context, marker_ids, marker_mask)
            state = state + (1.0 / steps) * velocity
        return state
