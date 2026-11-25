"""Registered H&E/protein tiles in a per-slide NumPy manifest."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

IMAGE_SIZE = (256, 256)


@dataclass(frozen=True)
class Slide:
    key: str
    patient: str
    split: str
    markers: tuple[str, ...]
    he: Path
    proteins: Path


def read_manifest(path: str | Path) -> tuple[str, list[Slide]]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    slides = [
        Slide(
            row["sample_key"],
            row["patient"],
            row["split"],
            tuple(row["markers"]),
            path.parent / row["he"],
            path.parent / row["proteins"],
        )
        for row in manifest["slides"]
    ]
    if len({slide.key for slide in slides}) != len(slides):
        raise ValueError("Every slide must have a unique sample_key.")
    patients = {}
    for slide in slides:
        if slide.split not in {"train", "val", "test"}:
            raise ValueError(f"{slide.key}: split must be train, val, or test.")
        if patients.setdefault(slide.patient, slide.split) != slide.split:
            raise ValueError(f"Patient {slide.patient} appears in multiple splits.")
        if not slide.markers or len(set(slide.markers)) != len(slide.markers):
            raise ValueError(f"{slide.key}: supply a nonempty list of unique markers.")
    return manifest["dataset"], slides


def resize_proteins(raw: torch.Tensor) -> torch.Tensor:
    if raw.shape[-2:] != IMAGE_SIZE:
        raw = F.interpolate(raw.float(), IMAGE_SIZE, mode="area")
    return raw.float().clamp_min(0)


def to_unit(raw: torch.Tensor, p99: torch.Tensor) -> torch.Tensor:
    upper = p99[:, None, None]
    image = resize_proteins(raw[None])[0]
    return (torch.minimum(image, upper).log1p() / upper.log1p()).clamp_(0, 1)


def from_unit(image: torch.Tensor, p99: torch.Tensor) -> torch.Tensor:
    """Map a unit image to clipped intensity units using fixed training p99."""
    return torch.expm1(image.float() * p99.float().log1p())


def read_thresholds(path: str | Path, dataset: str) -> dict[str, float]:
    document = json.loads(Path(path).read_text())
    if document["dataset"] != dataset:
        raise ValueError("The p99 file must belong to the requested dataset.")
    return document["p99"]


class PairedTiles(Dataset):
    def __init__(
        self,
        slides: list[Slide],
        split: str,
        marker_keys: list[str],
        p99: dict[str, float],
    ):
        self.slides = [slide for slide in slides if slide.split == split]
        self.round_to_fp16 = split in {"train", "val"}
        self.ids = {key: i for i, key in enumerate(marker_keys)}
        self.p99 = p99
        self.arrays = {}
        self.indices = []
        self.panels = []
        for slide_index, slide in enumerate(self.slides):
            channels = sorted(range(len(slide.markers)), key=slide.markers.__getitem__)
            keys = [slide.markers[i] for i in channels]
            ids = torch.tensor([self.ids[key] for key in keys], dtype=torch.long)
            thresholds = torch.tensor([p99[key] for key in keys], dtype=torch.float32)
            if not (thresholds.isfinite() & (thresholds > 0)).all():
                raise ValueError(f"{slide.key}: training p99 must be finite and positive.")
            self.panels.append((channels, ids, thresholds))
            he = np.load(slide.he, mmap_mode="r")
            protein = np.load(slide.proteins, mmap_mode="r")
            if he.ndim != 4 or he.shape[1] != 3 or he.dtype != np.uint8:
                raise ValueError(f"{slide.he}: expected uint8 [N, 3, H, W].")
            if protein.ndim != 4 or protein.shape[:2] != (len(he), len(channels)):
                raise ValueError(f"{slide.proteins}: expected [N, K, H, W] aligned with H&E.")
            self.indices.extend((slide_index, index) for index in range(len(he)))
        if not self.indices:
            raise ValueError(f"No tiles in the {split} split.")

    def __len__(self) -> int:
        return len(self.indices)

    def pair_count(self, indices) -> int:
        return sum(len(self.panels[self.indices[i][0]][1]) for i in indices)

    def __getitem__(self, index):
        # A distributed training tail can carry duplicate, zero-loss tiles.
        index, real = (index, True) if isinstance(index, int) else index
        slide_index, patch_index = self.indices[index]
        slide = self.slides[slide_index]
        if slide_index not in self.arrays:
            self.arrays[slide_index] = (
                np.load(slide.he, mmap_mode="r"),
                np.load(slide.proteins, mmap_mode="r"),
            )
        he, proteins = self.arrays[slide_index]
        channels, ids, thresholds = self.panels[slide_index]
        target = to_unit(torch.from_numpy(proteins[patch_index, channels]), thresholds)
        if self.round_to_fp16:
            target = target.to(torch.float16)
        return {
            "he": torch.from_numpy(he[patch_index].copy()),
            "target": target,
            "marker_ids": ids,
            "sample_key": slide.key,
            "patient": slide.patient,
            "patch_index": patch_index,
            "real": real,
        }


def collate_panels(items: list[dict]) -> dict:
    """Pad only the marker axis and mark which entries are measured."""
    width = max(len(item["marker_ids"]) for item in items)
    ids = torch.zeros((len(items), width), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    target = torch.zeros((len(items), width, *IMAGE_SIZE), dtype=items[0]["target"].dtype)
    for row, item in enumerate(items):
        count = len(item["marker_ids"])
        ids[row, :count] = item["marker_ids"]
        mask[row, :count] = True
        target[row, :count] = item["target"]
    he = [item["he"] for item in items]
    if any(tile.shape != he[0].shape for tile in he):
        raise ValueError("H&E tiles in a batch must have the same pixel dimensions.")
    return {
        "he": torch.stack(he),
        "target": target,
        "marker_ids": ids,
        "marker_mask": mask,
        "real": torch.tensor([item["real"] for item in items]),
        "metadata": [(item["sample_key"], item["patient"], item["patch_index"]) for item in items],
    }
