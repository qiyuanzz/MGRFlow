"""Frozen H-optimus-0 and Stable Diffusion 2.1 protein VAE."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import flow_precision

HE_MEAN = (0.707223, 0.578729, 0.703617)
HE_STD = (0.211883, 0.230117, 0.177517)
HE_BLOCKS = (9, 19, 29, 39)


class HOptimus(nn.Module):
    def __init__(self, model_dir: str | Path, device: torch.device):
        super().__init__()
        import timm

        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        self.encoder = timm.create_model(
            config["architecture"],
            pretrained=False,
            num_classes=config["num_classes"],
            global_pool=config["global_pool"],
            img_size=224,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        if (root / "model.safetensors").is_file():
            from safetensors.torch import load_file

            weights = load_file(root / "model.safetensors")
        else:
            weights = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(weights)
        self.encoder.requires_grad_(False).eval().to(device)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Extract [B, 4, 256, 1536] pre-final-norm spatial tokens from byte RGB."""
        image = rgb.float() / 255
        image = F.interpolate(
            image, size=(256, 256), mode="bicubic", align_corners=False, antialias=True
        ).clamp_(0, 1)
        mean = image.new_tensor(HE_MEAN)[None, :, None, None]
        std = image.new_tensor(HE_STD)[None, :, None, None]
        image = ((image - mean) / std).to(torch.float16).float()
        image = F.interpolate(
            image, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
        )
        with torch.autocast(rgb.device.type, dtype=torch.float16, enabled=rgb.is_cuda):
            levels = self.encoder.forward_intermediates(
                image, indices=HE_BLOCKS, norm=False, output_fmt="NLC", intermediates_only=True
            )
        features = torch.stack(levels, dim=1).to(torch.float16)
        return features if flow_precision(rgb.device) == torch.bfloat16 else features.float()


class ProteinVAE(nn.Module):
    def __init__(self, model_dir: str | Path, device: torch.device, dtype: torch.dtype):
        super().__init__()
        from diffusers import AutoencoderKL

        root = Path(model_dir)
        if (root / "vae").is_dir():
            root = root / "vae"
        self.vae = (
            AutoencoderKL.from_pretrained(str(root), local_files_only=True, torch_dtype=dtype)
            .requires_grad_(False)
            .eval()
            .to(device)
        )
        self.scale = float(self.vae.config.scaling_factor)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, pair_batch: int) -> torch.Tensor:
        """Encode [N, 256, 256] unit maps into [N, 4, 32, 32] latents."""
        latents = []
        for chunk in images.split(pair_batch):
            rgb = chunk[:, None].expand(-1, 3, -1, -1).contiguous()
            posterior = self.vae.encode(rgb.to(self.vae.dtype) * 2 - 1).latent_dist
            latents.append((posterior.mode() * self.scale).float())
        return torch.cat(latents)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, pair_batch: int) -> torch.Tensor:
        """Decode [N, 4, 32, 32] to [N, 256, 256] unit maps."""
        images = []
        for chunk in latents.split(pair_batch):
            rgb = self.vae.decode(chunk.to(self.vae.dtype) / self.scale).sample
            images.append(rgb.float().add_(1).mul_(0.5).clamp_(0, 1).mean(dim=1))
        return torch.cat(images)
