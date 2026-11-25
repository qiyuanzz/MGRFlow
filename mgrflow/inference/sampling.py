"""Deterministic panel generation from H&E and marker identities."""

import hashlib
import json

import torch

from ..model.flow import LATENT_SHAPE
from ..runtime import flow_precision


def identity_noise(
    dataset: str,
    metadata: list[tuple[str, str, int]],
    ids: torch.Tensor,
    marker_keys: list[str],
    seed: int,
) -> torch.Tensor:
    """Bind each noise draw to its patch and marker, independently of the batch."""
    panels = []
    for (sample, _, patch), panel in zip(metadata, ids.tolist(), strict=True):
        noise = []
        for marker in panel:
            key = json.dumps(
                [seed, dataset, sample, patch, marker_keys[marker]], separators=(",", ":")
            )
            digest = hashlib.sha256(key.encode()).digest()
            local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            generator = torch.Generator().manual_seed(local_seed)
            noise.append(torch.randn(LATENT_SHAPE, generator=generator))
        panels.append(torch.stack(noise))
    return torch.stack(panels)


@torch.no_grad()
def predict_panel(
    model,
    he_encoder,
    vae,
    batch: dict,
    *,
    dataset: str,
    marker_keys: list[str],
    steps: int,
    seed: int,
    device: torch.device,
    vae_batch: int,
) -> torch.Tensor:
    """Return decoded valid marker images in the row-major order of the mask."""
    noise = identity_noise(dataset, batch["metadata"], batch["marker_ids"], marker_keys, seed)
    full_batch = bool(batch["marker_mask"].all())
    mask = batch["marker_mask"].to(device)
    marker_ids = batch["marker_ids"].to(device)
    features = he_encoder(batch["he"].to(device))
    dtype = flow_precision(device)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
        latents = model.sample(
            noise.to(device),
            features,
            marker_ids,
            steps=steps,
            marker_mask=None if full_batch else mask,
        )
        return vae.decode(latents[mask], vae_batch)
