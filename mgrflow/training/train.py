"""Train MGRFlow with all measured markers for each H&E tile."""

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ..checkpoints import save_weights
from ..data.tiles import PairedTiles, collate_panels, read_manifest, read_thresholds
from ..evaluation.metrics import MarkerAverages, pearson
from ..inference.sampling import predict_panel
from ..model import MGRFlow, ModelConfig
from ..model.encoders import HOptimus, ProteinVAE
from ..model.flow import LATENT_SHAPE
from ..runtime import TrainingTiles, finalize, flow_precision, initialize, sum_ranks
from .ema import EMA


def marker_table(path: Path, use_smm: bool) -> tuple[list[str], torch.Tensor]:
    with np.load(path, allow_pickle=False) as table:
        keys = table["marker_keys"].astype(str).tolist()
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Supply a nonempty marker registry with unique keys.")
        if not use_smm:
            return keys, torch.empty(len(keys), 0)
        embeddings = torch.from_numpy(table["embeddings"]).float()
    if embeddings.shape != (len(keys), 640):
        raise ValueError("The ESM-2 table must contain one 640-vector per marker key.")
    return keys, embeddings


def cosine_schedule(optimizer, total_steps: int):
    warmup = max(1, int(0.05 * total_steps))

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def backward_flow(
    model, he_encoder, vae, batch, *, device, world, microbatch, vae_batch
) -> torch.Tensor:
    dtype = flow_precision(device)
    he = batch["he"].to(device, non_blocking=True)
    ids = batch["marker_ids"].to(device, non_blocking=True)
    mask = batch["marker_mask"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    real = mask & batch["real"].to(device)[:, None]
    pairs = sum_ranks(real.sum())

    with torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
        features = he_encoder(he)
        clean = torch.zeros((*ids.shape, *LATENT_SHAPE), device=device)
        clean[mask] = vae.encode(target[mask].float(), vae_batch)
    source = torch.randn_like(clean)
    progress = torch.rand(len(he), device=device)
    t = progress[:, None, None, None, None]
    state = (1 - t) * source + t * clean
    velocity = clean - source

    total = torch.zeros((), device=device)
    for start in range(0, len(he), microbatch):
        end = min(start + microbatch, len(he))
        rows = slice(start, end)
        sync = model.no_sync() if world > 1 and end < len(he) else nullcontext()
        with sync:
            with torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
                predicted = model(
                    state[rows], progress[rows], features[rows], ids[rows], mask[rows]
                )
            error = (predicted.float() - velocity[rows]).square().mean(dim=(-3, -2, -1))
            # DDP averages gradients. Weight each real marker pair equally over all ranks.
            loss = (error * real[rows]).sum() * (world / pairs)
            loss.backward()
        total += loss.detach()
    return total


@torch.no_grad()
def validate(model, he, vae, loader, *, device, dataset, keys, seed, steps, vae_batch) -> float:
    metrics = MarkerAverages(keys, ("pearson",), device)
    model.eval()
    for batch in loader:
        prediction = predict_panel(
            model,
            he,
            vae,
            batch,
            dataset=dataset,
            marker_keys=keys,
            steps=steps,
            seed=seed,
            device=device,
            vae_batch=vae_batch,
        )
        mask = batch["marker_mask"]
        measured = batch["target"][mask].to(device)
        scores = pearson(prediction, measured)
        metrics.update(batch["marker_ids"][mask].to(device), scores[:, None])
    value = metrics.result()["marker_macro"]["pearson"]
    if value is None:
        raise ValueError("All validation Pearson scores are undefined.")
    return value


def train(config: dict, directory: Path) -> None:
    device, rank, world = initialize(config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed + rank)
    model_config = ModelConfig(**config.get("model", {}))
    keys, embeddings = marker_table(directory / config["marker_table"], model_config.smm)
    dataset, slides = read_manifest(directory / config["manifest"])
    p99 = read_thresholds(directory / config["p99"], dataset)
    train_data = PairedTiles(slides, "train", keys, p99)
    val_data = PairedTiles(slides, "val", keys, p99)
    loader_options = {
        "collate_fn": collate_panels,
        "num_workers": config.get("workers", 2),
        "pin_memory": device.type == "cuda",
    }
    batch_size = int(config.get("global_batch", 32))
    if batch_size < world or batch_size % world:
        raise ValueError("global_batch must be a positive multiple of the number of ranks.")
    sampler = TrainingTiles(len(train_data), rank, world, seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size // world, sampler=sampler, **loader_options
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.get("validation_batch", 1),
        sampler=range(rank, len(val_data), world),
        **loader_options,
    )
    he = HOptimus(directory / config["hoptimus"], device)
    vae = ProteinVAE(directory / config["vae"], device, flow_precision(device))
    model = MGRFlow(embeddings, model_config).to(device)
    parallel = model
    if world > 1:
        device_ids = [device.index] if device.type == "cuda" else None
        parallel = DistributedDataParallel(model, device_ids=device_ids, broadcast_buffers=False)
    ema = EMA(model, config.get("ema_decay", 0.9999)) if config.get("ema", True) else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get("learning_rate", 1e-4), weight_decay=0.01
    )
    epochs = int(config.get("epochs", 50))
    scheduler = cosine_schedule(optimizer, epochs * len(train_loader))
    output = directory / config["output"]
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    best, stale = -math.inf, 0
    patience = config.get("early_stop_patience", 5)
    microbatch = config.get("flow_microbatch", 1)
    vae_batch = config.get("vae_batch", 8)

    for epoch in range(1, epochs + 1):
        sampler.epoch = epoch - 1
        parallel.train()
        running = torch.zeros((), device=device)
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            running += backward_flow(
                parallel,
                he,
                vae,
                batch,
                device=device,
                world=world,
                microbatch=microbatch,
                vae_batch=vae_batch,
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            if step % 100 == 0:
                loss = float(sum_ranks(running.clone()) / (world * step))
                if rank == 0:
                    print(
                        f"epoch {epoch} | step {step}/{len(train_loader)} | loss {loss:.5f}",
                        flush=True,
                    )

        selected = ema.model if ema is not None else model
        score = validate(
            selected,
            he,
            vae,
            val_loader,
            device=device,
            dataset=dataset,
            keys=keys,
            seed=seed,
            steps=config.get("validation_steps", 1),
            vae_batch=vae_batch,
        )
        loss = float(sum_ranks(running) / (world * len(train_loader)))
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        if rank == 0:
            print(f"epoch {epoch} | loss {loss:.5f} | val Pearson {score:.5f}", flush=True)
            save_weights(
                output / "last_online.pt",
                model,
                keys,
                dataset,
                p99,
                epoch=epoch,
                weight_source="online",
            )
            if improved:
                save_weights(
                    output / "best.pt",
                    selected,
                    keys,
                    dataset,
                    p99,
                    epoch=epoch,
                    val_pearson=score,
                    weight_source="ema" if ema is not None else "online",
                    ema_decay=ema.decay if ema is not None else None,
                )
        if patience and stale >= patience:
            break
    finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.ema is not None:
        config["ema"] = args.ema
    train(config, args.config.resolve().parent)
