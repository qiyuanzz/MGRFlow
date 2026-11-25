"""Evaluate paired test tiles and write patch-level image metrics."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .checkpoints import load_weights
from .data import PairedTiles, collate_panels, read_manifest, read_thresholds
from .encoders import HOptimus, ProteinVAE
from .fid import MarkerFID
from .generate import predict_panel
from .metrics import IMAGE_METRICS, MarkerAverages, patch_scores
from .runtime import finalize, flow_precision, initialize


class PatchResults:
    def __init__(self, table, path: Path, rank: int, features: bool, pairs: int):
        self.csv = csv.writer(table)
        self.csv.writerow(
            [
                "dataset",
                "sample_key",
                "patient",
                "patch_index",
                "marker_key",
                "feature_row",
                *IMAGE_METRICS,
            ]
        )
        self.features = []
        if features:
            for kind in ("predicted", "measured"):
                self.features.append(
                    np.lib.format.open_memmap(
                        path / f"fid_{kind}_rank{rank}.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(pairs, 2048),
                    )
                )
        self.rows = 0

    def write(self, rows: list, scores: torch.Tensor, embeddings: tuple | None) -> None:
        for keys, values in zip(rows, scores.cpu().tolist(), strict=True):
            row = self.rows if self.features else ""
            self.csv.writerow(
                [*keys, row, *("" if math.isnan(value) else value for value in values)]
            )
            self.rows += 1
        if self.features:
            start = self.rows - len(rows)
            for array, value in zip(self.features, embeddings, strict=True):
                array[start : self.rows] = value.cpu().numpy()

    def flush(self) -> None:
        for array in self.features:
            array.flush()


@torch.no_grad()
def evaluate_budget(model, he, vae, loader, path, *, args, dataset, keys, device, rank, steps):
    metrics = MarkerAverages(keys, IMAGE_METRICS, device)
    fid = MarkerFID(keys, device) if args.fid or args.save_fid_features else None
    total = loader.dataset.pair_count(loader.sampler)
    if rank == 0:
        path.mkdir(parents=True)
    if dist.is_initialized():
        dist.barrier()
    with gzip.open(path / f"patch_metrics_rank{rank}.csv.gz", "wt", newline="") as file:
        results = PatchResults(file, path, rank, args.save_fid_features, total)
        for index, batch in enumerate(loader, 1):
            prediction = predict_panel(
                model,
                he,
                vae,
                batch,
                dataset=dataset,
                marker_keys=keys,
                steps=steps,
                seed=args.seed,
                device=device,
                vae_batch=args.vae_batch,
            )
            mask = batch["marker_mask"]
            measured = batch["target"][mask].to(device)
            ids = batch["marker_ids"][mask].to(device)
            rows = [
                (dataset, sample, patient, patch, keys[marker])
                for (sample, patient, patch), panel, valid in zip(
                    batch["metadata"], batch["marker_ids"], mask, strict=True
                )
                for marker in panel[valid].tolist()
            ]
            for start in range(0, len(rows), args.metric_batch):
                stop = start + args.metric_batch
                x, y = prediction[start:stop], measured[start:stop]
                scores = patch_scores(x, y)
                markers = ids[start:stop]
                metrics.update(markers, scores)
                embeddings = None
                if fid is not None:
                    features = fid.features(torch.cat((x, y))).chunk(2)
                    fid.update(markers, *features)
                    embeddings = features
                results.write(rows[start:stop], scores, embeddings)
            if rank == 0 and index % 100 == 0:
                print(f"{steps} steps | batch {index}/{len(loader)}", flush=True)
        if results.rows != total:
            raise RuntimeError("Patch output count does not match the evaluation split.")
        results.flush()
    report = metrics.result()
    if fid is not None:
        scores = fid.result(rank)
        if rank == 0:
            for key, value in scores.items():
                report["per_marker"][key]["fid"] = value
            values = [value for value in scores.values() if value is not None]
            report["marker_macro"]["fid"] = sum(values) / len(values) if values else None
    if rank == 0:
        report |= {"dataset": dataset, "split": args.split, "steps": steps, "seed": args.seed}
        (path / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            f"{steps} steps | {report['evaluated_pairs']} pairs | {report['marker_macro']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hoptimus", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--p99", type=Path, help="Override with fixed training thresholds for this dataset."
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--steps", type=int, nargs="+", default=[4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="H&E tiles per process.")
    parser.add_argument("--vae-batch", type=int, default=8)
    parser.add_argument("--metric-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fid", action="store_true")
    parser.add_argument(
        "--save-fid-features", action="store_true", help="Save FID features alongside patch scores."
    )
    args = parser.parse_args()
    if min(args.steps) < 1 or len(set(args.steps)) != len(args.steps):
        parser.error("--steps must contain distinct positive integers.")
    if min(args.batch_size, args.vae_batch, args.metric_batch) < 1:
        parser.error("Batch sizes must be positive.")
    device, rank, world = initialize(args.device)
    model, info = load_weights(args.checkpoint, device)
    dataset, slides = read_manifest(args.manifest)
    keys = info["marker_keys"]
    p99 = read_thresholds(args.p99, dataset) if args.p99 else info.get("p99")
    if p99 is None or (dataset != info["dataset"] and args.p99 is None):
        parser.error("--p99 must supply the fixed training thresholds for this dataset.")
    data = PairedTiles(slides, args.split, keys, p99)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=range(rank, len(data), world),
        collate_fn=collate_panels,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    he = HOptimus(args.hoptimus, device)
    vae = ProteinVAE(args.vae, device, flow_precision(device))
    for steps in args.steps:
        evaluate_budget(
            model,
            he,
            vae,
            loader,
            args.output / f"steps_{steps:02d}",
            args=args,
            dataset=dataset,
            keys=keys,
            device=device,
            rank=rank,
            steps=steps,
        )
    finalize()


if __name__ == "__main__":
    main()
