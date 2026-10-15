# Quickstart: Reproducible Legacy Lesson Evidence

## Requirements

- Python 3.10–3.13 and the project dependencies.
- PyTorch and TorchVision installed as described in the project README.
- Gymnasium for notebooks 05–08.
- Network access on the first run of notebook 04 if CIFAR-10 or the pinned `timm/resnet18.a1_in1k` weights are not cached.
- CPU is sufficient. No robot or private data is required.

## Run

1. Start Jupyter Lab from the repository root.
2. Run notebooks 01 and 04–08 from a clean kernel.
3. Run notebook 05 before notebook 06 in the same working directory. Notebook 05 writes an ignored checkpoint that notebook 06 reads.
4. Check that notebooks 05 and 06 report returns for every declared evaluation seed and use the same seed list for each compared method.
5. Check that notebook 08 labels its eight settings as single-seed exploratory results.

## Expected output

Each notebook completes and reports its result, device, and run settings. The final report is `docs/evidence/legacy-lessons-local-run.md`. Dataset files and checkpoints remain in ignored paths. Paper benchmark results remain in `docs/REPRODUCTION_MATRIX.md`.
