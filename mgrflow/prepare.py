"""Fit and export the fixed training-set p99 for each marker."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import read_manifest, resize_proteins

BUCKETS = 1 << 16


def pixel_batches(slides: list, tile_batch: int):
    for slide in slides:
        if slide.split != "train":
            continue
        images = np.load(slide.proteins, mmap_mode="r")
        for start in range(0, len(images), tile_batch):
            batch = torch.from_numpy(images[start : start + tile_batch].copy())
            if not torch.isfinite(batch.float()).all():
                raise ValueError(f"{slide.key}: measured training intensities must be finite.")
            batch = resize_proteins(batch).numpy()
            for i, key in enumerate(slide.markers):
                yield key, np.ascontiguousarray(batch[:, i] + np.float32(0)).view(np.uint32).ravel()


def fit_p99(slides: list, tile_batch: int) -> dict[str, float]:
    """Exact linear percentile via two streaming histograms of float32 bits."""
    # Unsigned float32 bit patterns preserve the order of nonnegative pixels.
    coarse = {}
    for key, bits in pixel_batches(slides, tile_batch):
        counts = np.bincount(bits >> 16, minlength=BUCKETS)
        if key in coarse:
            coarse[key] += counts
        else:
            coarse[key] = counts

    plans = {}
    fine = {}
    for key, hist in coarse.items():
        cumulative = hist.cumsum()
        position = 0.99 * (int(cumulative[-1]) - 1)
        ranks = np.array([np.floor(position), np.ceil(position)], dtype=np.int64)
        buckets = np.searchsorted(cumulative, ranks, side="right")
        before = np.where(buckets > 0, cumulative[buckets - 1], 0)
        plans[key] = (buckets, ranks - before, position - ranks[0])
        fine[key] = {bucket: np.zeros(BUCKETS, dtype=np.int64) for bucket in set(buckets)}

    for key, bits in pixel_batches(slides, tile_batch):
        high = bits >> 16
        for bucket, hist in fine[key].items():
            hist += np.bincount(bits[high == bucket] & 0xFFFF, minlength=BUCKETS)

    result = {}
    for key, (buckets, ranks, fraction) in plans.items():
        values = []
        for bucket, rank in zip(buckets, ranks, strict=True):
            low = np.searchsorted(fine[key][bucket].cumsum(), rank, side="right")
            values.append(float(np.uint32((int(bucket) << 16) | int(low)).view(np.float32)))
        result[key] = values[0] + (values[1] - values[0]) * float(fraction)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile-batch", type=int, default=8)
    args = parser.parse_args()
    dataset, slides = read_manifest(args.manifest)
    p99 = fit_p99(slides, args.tile_batch)
    required = {key for slide in slides for key in slide.markers}
    missing = sorted(key for key in required if p99.get(key, 0) <= 0)
    if missing:
        raise ValueError(f"These markers have no positive training p99: {', '.join(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dataset": dataset, "p99": p99}, indent=2) + "\n")


if __name__ == "__main__":
    main()
