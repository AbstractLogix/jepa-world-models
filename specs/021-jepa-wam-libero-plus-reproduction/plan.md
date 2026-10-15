# Implementation Plan: JEPA-WAM LIBERO-Plus Checkpoint Re-evaluation

**Branch**: 021-jepa-wam-libero-plus-reproduction | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

## Summary

Re-evaluate the released JEPA-WAM checkpoint on the authors' documented LIBERO-Plus spatial-suite path. Start with the upstream one-task, one-action diagnostic. Run the full 2,402-task suite only after the smoke succeeds and the measured runtime and memory fit the available host. Do not repeat training.

## Technical Context

- **Language/Version**: Python 3.10.16 per the released environment.
- **Primary Dependencies**: Pinned JEPA-WAM source and its Python 3.10/CUDA 12.1 environment; PyTorch GPU support must be qualified on the RTX 5070.
- **Storage**: Download only the evaluation checkpoints, language/vision encoders, LIBERO-Plus source, and simulator assets. Do not download RLDS training data or the full checkpoint repository.
- **Evaluation**: Upstream LIBERO-Plus evaluator, one trial per task, all seven categories.
- **Target Platform**: One local RTX 5070 with 12 GB VRAM and MuJoCo/EGL. Neither the runtime nor renderer is qualified yet.
- **Project Type**: Released-checkpoint evaluation and evidence documentation.
- **Constraints**: Keep all third-party assets and temporary source outside Git. Review the pinned checkpoint-loading path before loading any serialized weights. Do not describe the result as a training reproduction.

## Constitution Check

- Preserve source and artifact provenance: PASS. Pin code, checkpoint, benchmark, and required files.
- Label evidence honestly: PASS. Separate a pipeline smoke, checkpoint evaluation, and training reproduction.
- Make the run reproducible: PASS. Preserve task outcomes, category summaries, environment, and compute.
- Keep changes small: PASS. Use the upstream evaluator; add no local model implementation.

## Project Structure

Feature artifacts are in this directory. The preflight and any eventual evaluation record are in docs/reproductions/jepa-wam-libero-plus.md. Model code, benchmark assets, and raw outputs stay in a disposable external checkout.

**Structure Decision**: Keep model code and benchmark evaluation in a disposable external checkout. Keep the source review, scope, and eventual result record in this project.
