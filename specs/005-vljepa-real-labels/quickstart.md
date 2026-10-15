# Quickstart: VL-JEPA Real Label Embeddings

## Requirements

- Python 3.10–3.13 and the project environment.
- PyTorch and TorchVision installed as described in the root README.
- Network access on the first run. The notebook downloads up to about 300 MB: CIFAR-10 (163 MB), ImageNet ResNet-18 weights (44.7 MB), and MiniLM weights (90.9 MB), plus small metadata files.
- About 2,000 CIFAR-10 images are resized for the frozen image encoder. CPU is supported; GPU is optional.
- CIFAR-10's source page requests a citation but does not show dataset license terms. TorchVision says model weights can have separate terms. Review applicable terms before use and do not redistribute downloaded files.

## Install

```bash
uv sync --locked --extra vljepa
```

The lock supplies CPU PyTorch wheels. For CUDA 12.8, install the matching wheels after sync:

```bash
uv pip install -r requirements-torch.txt  # optional CUDA 12.8 override
```

For a supported GPU, follow the root README's matching PyTorch wheel instructions instead of the CPU wheel command.

## Run

Start Jupyter from the repository root and run `notebooks/11_vl_jepa.ipynb` from top to bottom. The first run needs network access. Later offline runs need all three assets already cached.

## Expected output

- A source and resource summary before model and data loading.
- Ten normalized class-prompt embeddings and a balanced sample-count check.
- Five runs for the text-embedding predictor and five runs for the linear classifier, using identical image examples and visual features.
- Per-seed, mean, median, standard deviation, uncertain 95% intervals across five seeds, paired per-image accuracy difference with a class-stratified bootstrap interval, class-wise accuracy, and candidate prompt scores for held-out images.
- Explicit limits: small CIFAR-10 labels, supervised closed-set learning, ImageNet visual pretraining, no paper checkpoint, and no VL-JEPA reproduction.
