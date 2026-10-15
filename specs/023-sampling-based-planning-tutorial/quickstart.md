# Quickstart: Sampling-Based Planning Tutorial

## Requirements

- Use the repository's locked `uv` environment and a fresh Jupyter CPU kernel.
- No dataset, checkpoint, GPU, simulator, robot, or new Python package is required.

## Run

1. Sync the locked project environment with `uv sync --locked` if it is not already installed.
2. Open [`notebooks/21_sampling_based_planning.ipynb`](../../notebooks/21_sampling_based_planning.ipynb).
3. Select a fresh CPU kernel and run every cell from top to bottom.
4. Compare the regenerated results with [`docs/reproductions/sampling-based-planning.md`](../../docs/reproductions/sampling-based-planning.md) and the claim limits in the final notebook cells.

The notebook uses fixed seeds and keeps episode records in memory. Its output is an illustrative state-space analogy. It does not reproduce the cited paper.
