# Implementation Plan: Paper-aligned JEPA objectives

**Branch**: `007-paper-aligned-losses` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Project vision task T014 in `specs/001-project-vision/tasks.md`.

## Summary

Compare the current teaching losses with the core objectives from LeJEPA and Rectified LpJEPA. Keep the existing proxy functions available. Add exact paper-objective helpers with seeded random projections and target sampling. Run matched, small CIFAR-10 studies in notebooks 02 and 03, then report all five seeds and the limits. These runs validate the local code path and objective behavior. They do not reproduce the papers' ImageNet benchmarks.

## Technical Context

**Language/Version**: Python 3.10–3.13.

**Primary Dependencies**: Existing PyTorch, TorchVision, NumPy, scikit-learn, and Jupyter environment. No new dependency.

**Storage**: Existing local CIFAR-10 cache under `notebooks/data`. No checkpoints or generated run output are committed.

**Testing**: Notebook source and JSON checks, loss-function invariants in the notebooks, and complete CPU execution of notebooks 02 and 03 to temporary output files.

**Target Platform**: CPU on the project-supported local Python environment.

**Project Type**: Educational notebooks with a small shared helper module.

**Performance Goals**: Finish the planned CPU comparison on 640 unlabeled images, five seeds, and 20 epochs per arm without loading paper-scale models or data.

**Constraints**: Keep data splits fixed and disjoint; seed all model, projection, and target-sample randomness; use the project evaluation protocol; distinguish local measurements from paper results; disclose the smaller network, dataset, optimizer, epoch count, projection counts, and evaluation subset.

**Scale/Scope**: One new objective helper module, two existing notebooks, one run report, the curriculum, roadmap, reproduction matrix, and linked project-vision status.

## Constitution Check

- **Teach from evidence**: Pass. Cite both primary papers, first-party code, and the official CIFAR-10 source. Keep local and paper-reported scores in separate records.
- **Label teaching examples honestly**: Pass. Label these runs as small-scale objective studies, not paper reproductions.
- **Make the learning path usable**: Pass. Use the existing local PyTorch environment and CIFAR-10 path; state data and compute needs before execution.
- **Measure before making performance claims**: Pass. Use fixed splits, paired seeds, a named probe and sparsity metrics, and show every run.
- **Keep changes small and reviewable**: Pass. Reuse existing dependencies and proxy helpers; add one focused module and avoid a broader model refactor.

## Design Decisions

1. Keep `sigreg_loss` and `rectified_distribution_matching_loss` as teaching proxies. Add separately named paper-aligned functions so existing lessons and code do not silently change meaning.
2. Implement the LeJEPA Epps-Pulley statistic from the paper equation: normalized Gaussian directions, empirical characteristic functions, Gaussian target characteristic function, Gaussian weighting, and trapezoidal integration.
3. Implement Rectified LpJEPA RDMReg for the paper's default `p=1`, `mu=0` Rectified Laplace target. Sample target values and random directions from explicit generators, project both batches, sort along the batch axis, and compute mean squared distance.
4. Use the same small CIFAR-10 subsets, four-stage CNN backbone, and training budget within each pair. Notebook 03 adds the paper-required three-layer projector and final ReLU to both arms; its teaching proxy remains the existing variance/covariance and L1/entropy objective.
5. Raise the shared frozen-probe iteration limit after the initial run showed convergence warnings. Use five fixed seeds and the project evaluation protocol. Report t-intervals as uncertain because five runs give limited precision.
6. Keep the actual ImageNet-100 recipes as future reproduction work. The local CIFAR-10 study changes dataset, model, image size, optimizer, epochs, and probe size.

## Project Structure

```text
src/jepa_tutorial/paper_losses.py
src/jepa_tutorial/eval_utils.py
notebooks/02_lejepa_from_scratch.ipynb
notebooks/03_rectified_lpjepa.ipynb
docs/reproductions/lejepa-rectified-lpjepa.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
docs/REPRODUCTION_MATRIX.md
specs/001-project-vision/tasks.md
specs/001-project-vision/alignment.md
specs/001-project-vision/remaining-work.md
specs/007-paper-aligned-losses/
```

**Structure Decision**: Add paper-objective functions under the existing `src/jepa_tutorial` package and teach their use in notebooks 02 and 03. Keep experiment setup and results in those notebooks and the run report.

## Complexity Tracking

No constitution exceptions or new packages are required.
