# Quickstart: Qantara Inference Lesson

## Local lesson

1. Open `notebooks/18_qantara_three_inference_modes.ipynb` in Jupyter.
2. Run every cell in a fresh CPU kernel.
3. Read the notebook's label for each toy calculation before comparing it with the paper.

The local lesson must not need external model weights, Push-T data, a GPU, or new dependencies.

## Optional paper evaluation

The Push-T checkpoint run is not ready to start. Before any download:

1. Verify that the selected HDF5 dataset is the paper's DINO-WM corpus and record its episode manifest.
2. Review the pinned checkpoint loader and serialized-object behavior. Define a controlled loading procedure before loading any full checkpoint.
3. Recheck source revisions, terms, sizes, and hashes. Estimate expanded size and confirm available storage before transfer.
4. Qualify the official Python, PyTorch, stable-worldmodel, MuJoCo, CUDA, and headless renderer versions on a supported GPU. Record software, hardware, time, and peak memory.
5. Pass one headless rendering smoke before transferring the full data or starting evaluation.
6. Declare five evaluation seeds before running. The paper does not name its exact five seeds; a local set must be reported as a protocol replication.
7. Preserve raw episode rows and environment records outside Git.

The current host fails the renderer gate. See [the preflight report](../../docs/reproductions/qantara-preflight.md). Do not start the official evaluation until the gate passes.
