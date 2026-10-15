# Quickstart: Paper-aligned JEPA objectives

## Requirements

- Python 3.10–3.13 and the repository's PyTorch/TorchVision notebook environment.
- About 163 MB for CIFAR-10 if the data is not already cached. Keep it under `notebooks/data`; do not commit it.
- CPU is supported. On the measured WSL2 CPU, model-run timers summed to about 16 minutes for notebook 02 and 4 minutes for notebook 03, plus notebook startup and data loading. Other hosts may take longer.
- The self-supervised subset has 640 images (64 per class). The initial 1,000-image attempt exceeded its 15-minute cell limit; all other study settings remain fixed.

## Run

1. Start Jupyter from the project environment.
2. Open `notebooks/02_lejepa_from_scratch.ipynb` and run all cells from a clean kernel.
3. Open `notebooks/03_rectified_lpjepa.ipynb` and run all cells from a clean kernel.
4. Compare each seed with [the run report](../../docs/reproductions/lejepa-rectified-lpjepa.md). The notebooks show each run, mean, median, and uncertain 95% interval.

To execute without writing outputs into the source notebooks, use Jupyter's `nbconvert` with a temporary output directory. Keep the project root as the kernel working directory so the notebook resolves `notebooks/data`.

## Expected result

Each notebook prints its dataset and model sizes, software versions, device, method settings, per-seed result, aggregate metrics, and failures. The run is a small local objective study. It does not reproduce the papers' ImageNet-1K or ImageNet-100 results.
