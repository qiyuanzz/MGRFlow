# MGRFlow

## Installation

Requires Python 3.10 or newer. Run from the repository root.

```bash
pip install -r requirements.txt
```

## Training

Prepare paired H&E and protein tiles using the [example manifest](configs/manifest.example.json). Set the dataset and local H-optimus-0 and SD 2.1-base VAE paths in a copy of the [training config](configs/train.yaml).

```bash
cp configs/train.yaml configs/train.local.yaml
mgrflow-fit-p99 data/manifest.json data/p99.json
mgrflow-embed data/marker_sequences.json data/markers.npz
torchrun --standalone --nproc_per_node=4 -m mgrflow.train --config configs/train.local.yaml
```

The embedding command takes a JSON map from marker keys to amino-acid sequences. For single-GPU training, use `mgrflow-train --config configs/train.local.yaml`. EMA is enabled by default. Pass `--no-ema` to disable it.

## Testing

```bash
mgrflow-evaluate \
  --manifest data/manifest.json --checkpoint runs/mgrflow/best.pt \
  --hoptimus weights/H-optimus-0 --vae weights/sd-2-1-base/vae \
  --steps 4 --fid --output runs/mgrflow/test
```

Use `--steps 1 2 4 8 16` for a sweep. Each budget writes a summary and patch-level metrics.
