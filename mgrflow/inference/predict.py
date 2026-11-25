"""Predict a requested panel for a single RGB H&E tile."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..checkpoints import load_weights
from ..data.tiles import from_unit, read_thresholds
from ..model.encoders import HOptimus, ProteinVAE
from ..runtime import flow_precision, initialize
from .sampling import predict_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hoptimus", required=True, type=Path)
    parser.add_argument("--vae", required=True, type=Path)
    parser.add_argument("--markers", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vae-batch", type=int, default=8)
    parser.add_argument("--save-intensity", action="store_true")
    parser.add_argument(
        "--p99", type=Path, help="Required for intensity outputs with original exports."
    )
    args = parser.parse_args()
    device, _, world = initialize(args.device)
    if world != 1:
        raise ValueError("Run single-tile prediction with one process.")
    model, info = load_weights(args.checkpoint, device)
    keys = info["marker_keys"]
    if len(set(args.markers)) != len(args.markers) or any(key not in keys for key in args.markers):
        raise ValueError(f"Request unique keys from the checkpoint registry: {', '.join(keys)}")
    ordered = sorted(args.markers)
    ids = torch.tensor([[keys.index(key) for key in ordered]])
    with Image.open(args.image) as image:
        he = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None]
    batch = {
        "he": he,
        "marker_ids": ids,
        "marker_mask": torch.ones_like(ids, dtype=torch.bool),
        "metadata": [(args.image.stem, "", 0)],
    }
    he_encoder = HOptimus(args.hoptimus, device)
    vae = ProteinVAE(args.vae, device, flow_precision(device))
    outputs = predict_panel(
        model,
        he_encoder,
        vae,
        batch,
        dataset=info["dataset"],
        marker_keys=keys,
        steps=args.steps,
        seed=args.seed,
        device=device,
        vae_batch=args.vae_batch,
    )
    images = outputs.cpu()[[ordered.index(key) for key in args.markers]]
    result = {"marker_keys": np.array(args.markers), "unit": images.numpy()}
    if args.save_intensity:
        p99 = read_thresholds(args.p99, info["dataset"]) if args.p99 else info.get("p99")
        if p99 is None:
            parser.error(
                "--save-intensity requires training p99 values for the checkpoint dataset."
            )
        thresholds = torch.tensor([p99[key] for key in args.markers])[:, None, None]
        result["clipped_intensity"] = from_unit(images, thresholds).numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "panel.npz", **result)
    for index, image in enumerate(images):
        gray = (255 * image).round().to(torch.uint8).numpy()
        Image.fromarray(gray).save(args.output / f"marker_{index:02d}.png")


if __name__ == "__main__":
    main()
