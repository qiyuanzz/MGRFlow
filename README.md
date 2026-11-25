# MGRFlow

**Morphology-Grounded Relational Flow for Joint Multiplex Protein Prediction from Histology**

MGRFlow jointly predicts a requested panel of spatial protein maps from a single H&E tile. A frozen H-optimus-0 encoder supplies morphology, fixed ESM-2 embeddings specify the marker identities, and a Rectified Flow transformer transports a set of Gaussian latents to protein latents. A frozen Stable Diffusion 2.1-base VAE decodes the panel at 256 × 256 pixels.

## Model

| Paper module | Implementation | Role |
| --- | --- | --- |
| **SMM** · Sequence-conditioned Marker Modulation | [`smm.py`](mgrflow/smm.py) | Fuses a fixed 640-dimensional ESM-2 marker embedding with the shared flow-time embedding to condition AdaLN-Zero. |
| **MMA** · Multilevel Morphology Aggregation | [`mma.py`](mgrflow/mma.py) | Projects and fuses the aligned 16 × 16 grids from zero-indexed H-optimus-0 blocks 9, 19, 29, and 39. |
| **MIRA** · Morphology-Informed Relational Attention | [`mira.py`](mgrflow/mira.py) | Adds aligned morphology at the transformer input and exchanges information across markers at each position in every block. |

[`model.py`](mgrflow/model.py) connects the modules. Each marker's 4 × 32 × 32 VAE latent becomes 256 tokens through a lossless 2 × 2 rearrangement. In every block, the protein tokens first attend within their own spatial grid, then query the H&E grid, then attend across markers at the same position. An FFN completes the block. H&E tokens supply the keys and values for the cross-attention branch. Padding masks support batches with different numbers of measured markers.

The code uses progress `s` from noise (0) to data (1). With `s = 1 - t`, this is the time convention in the paper written in the direction of sampling. Training matches the velocity `z_data - z_noise` using MSE. Inference integrates the learned velocity using a configurable number of Euler steps.

## Install

Use Python 3.10 or newer and install the PyTorch build for your device. Training uses BF16 mixed precision on CUDA GPUs with BF16 support. CPU execution is suitable for small checks.

```bash
python -m pip install -e '.[esm,fid]'
```

The `esm` extra builds marker embeddings. The `fid` extra enables the Inception-based metric. Install with `python -m pip install -e .` when you already have a marker table and only need the paired image metrics.

Supply the pretrained assets in local directories.

- **H-optimus-0** requires `config.json` and `model.safetensors` or `pytorch_model.bin`. [H-optimus-0 model card](https://huggingface.co/bioptimus/H-optimus-0)
- **Stable Diffusion 2.1-base VAE** requires the Diffusers `vae/` directory with `config.json` and its model weights. [Diffusers AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)
- **ESM-2 150M** can be used through its model ID or a local Transformers directory. [ESM-2 model card](https://huggingface.co/biohub/esm2_t30_150M_UR50D)

## Prepare registered tiles

Store one JSON manifest per cohort. Each slide references aligned H&E and raw measured protein arrays. Each patient or donor belongs to exactly one of `train`, `val`, or `test`. Paths are relative to the manifest. Add one entry for each slide.

```json
{
  "dataset": "example_cohort",
  "slides": [
    {
      "sample_key": "slide_001",
      "patient": "patient_001",
      "split": "train",
      "he": "slide_001/he.npy",
      "proteins": "slide_001/proteins.npy",
      "markers": ["CD3", "CD8"]
    }
  ]
}
```

- `he.npy` holds uint8 RGB tiles with shape `[N, 3, H, W]`.
- `proteins.npy` holds raw intensities with shape `[N, K, H, W]` in the declared marker order. Pair array rows by tissue location. H&E and protein tiles cover the same physical field of view.
- Use only the measured markers for each slide. Slides can have different panels. Keep marker keys consistent across the manifest, marker table, and checkpoint.

Fit each marker's 99th percentile from all area-resized training pixels. The fitter stores a single threshold per dataset and marker. Training, validation, and test use these same thresholds.

```bash
mgrflow-fit-p99 data/manifest.json data/p99.json
```

Protein targets are 256 × 256 maps with values `log1p(clip(I, 0, p99)) / log1p(p99)`. The preprocessing consists of area resizing, clipping, and log scaling. The VAE encodes these unit-range maps as grayscale replicated to three channels, using its posterior mode. Supply registered tiles and any dataset-specific channel exclusions when building the manifest.

With SMM enabled, create a JSON object mapping each marker key to its amino-acid sequence. A marker targeting multiple proteins can map to a list of sequences. The embedding command averages ESM-2 residue embeddings for each sequence, then averages the sequence vectors for a multi-target marker.

```bash
mgrflow-embed data/marker_sequences.json data/markers.npz
```

You can also supply your own NumPy archive with `marker_keys` (1D string array) and `embeddings` (FP32 array of shape `[number_of_markers, 640]`). When using one-hot marker conditioning, only `marker_keys` is required.

## Train

From the repository root, copy the [example config](configs/train.yaml). Set your asset and manifest paths in the local copy, which Git ignores.

```bash
cp configs/train.yaml configs/train.local.yaml
mgrflow-train --config configs/train.local.yaml
torchrun --standalone --nproc_per_node=4 -m mgrflow.train --config configs/train.local.yaml
```

The main model trains on all available markers jointly for each H&E tile. `global_batch` counts H&E tiles per optimizer update across GPUs. `flow_microbatch` splits the H&E batch for memory while keeping the marker panel together. `vae_batch` sets the number of marker images per frozen VAE call. Full validation runs at the end of each epoch. The highest validation marker-macro Pearson selects `best.pt`, and `last_online.pt` holds the latest online weights.

**EMA is enabled by default with decay 0.9999** and updates after each optimizer step. Validation and `best.pt` use the EMA weights. Set `ema: false` in the config or pass `--no-ema` to use online weights for both.

The template schedules up to 50 epochs and stops after five consecutive epochs without a higher validation Pearson. The original main comparison used selected checkpoints at epochs 50, 30, and 39 for HuBMAP-Placenta, CRC-CyCIF, and KPMP, respectively. Use `epochs`, `validation_steps`, and `early_stop_patience` to configure these settings for each run.

### Ablation switches

| Configuration | `model.smm` | `model.mma` | `model.mira` |
| --- | --- | --- | --- |
| One-hot identity and last-layer H&E baseline | false | false | false |
| + SMM | true | false | false |
| + MMA | true | true | false |
| + MIRA (MGRFlow) | true | true | true |

All four settings retain per-block H&E cross-attention. Disabling MIRA disables both the aligned input addition and the same-position cross-marker attention.

## Predict a panel

```bash
mgrflow-predict \
  --image data/example_he.png \
  --checkpoint runs/mgrflow/best.pt \
  --hoptimus weights/H-optimus-0 --vae weights/sd-2-1-base/vae \
  --markers CD3 CD8 --steps 4 --seed 42 --output predictions/example
```

`panel.npz` stores FP32 unit-range maps and the requested marker keys. The indexed grayscale PNGs follow the same order and share the fixed display range `[0, 1]`. Use `--save-intensity` to also export `expm1(unit * log1p(p99))` in the clipped intensity scale. The model needs H&E and marker identities for inference. The optional clipped-intensity export uses the fixed training thresholds saved in the checkpoint or supplied with `--p99`.

New checkpoints include the Flow weights, marker registry, and training p99 values. The loader also accepts the original native-256 full-model `best_full.pt` exports with four-branch AdaLN-Zero. Supply the cohort-specific `--p99` when evaluating or exporting clipped intensity from an original checkpoint. To see the exact marker keys for a checkpoint, load its metadata with `mgrflow.checkpoints.load_weights(path, torch.device("cpu"))` and inspect `metadata["marker_keys"]`.

## Evaluate

```bash
mgrflow-evaluate \
  --manifest data/manifest.json --checkpoint runs/mgrflow/best.pt \
  --hoptimus weights/H-optimus-0 --vae weights/sd-2-1-base/vae \
  --steps 1 4 --seed 42 --fid --output runs/mgrflow/test
```

For distributed evaluation, replace `mgrflow-evaluate` with `torchrun --standalone --nproc_per_node=4 -m mgrflow.evaluate`. Each step budget has its own result directory and starts from identical per-marker, per-tile noise. Reports are written after each budget. For a wider sweep use `--steps 1 2 4 8 16`.

- `summary.json` contains per-marker and equal-weight marker means. The four paired metrics are computed in FP32 on the unit-range maps.
- `patch_metrics_rank*.csv.gz` stores Pearson, SSIM, PSNR, and tie-aware Spearman for each patch and marker. Correlations for constant images have an empty CSV cell. Exact image matches have infinite PSNR.
- `--fid` computes distribution-level FID separately for each marker and averages those scores. It uses 2048-dimensional Inception pool-3 features of three-channel copies of the grayscale maps. `--save-fid-features` also writes predicted and measured feature arrays indexed by `feature_row` in each rank's CSV, for resampling at the patient or patch level. [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
