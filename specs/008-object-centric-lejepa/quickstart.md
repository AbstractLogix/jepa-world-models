# Quickstart: Object-centric LeJEPA Lesson

## Requirements

- Python 3.10–3.13 and the repository's notebook dependencies.
- PyTorch and TorchVision.
- About 800 MB for the Oxford-IIIT Pet images and annotations. The first run downloads the data to `notebooks/data/`.
- CPU works for the small model but can take longer. A CUDA device is optional.

Oxford-IIIT Pet is shared under CC BY-SA 4.0. Image copyright remains with the original owners. Keep downloaded files out of Git.

## Run

1. Open `notebooks/13_object_centric_lejepa.ipynb`.
2. Run every cell from a clean kernel.
3. The notebook uses a fixed balanced trainval subset, five seeds, two matched methods, and a frozen linear probe on the official test split.
4. Save run evidence in the notebook output and the generated report under `docs/reproductions/`. Do not commit raw images, masks, or checkpoints.

The local study is an exploratory transfer study. It does not reproduce the paper's COCO training or downstream benchmarks. See the [paper reproduction matrix](../../docs/REPRODUCTION_MATRIX.md) for exact-paper status.
