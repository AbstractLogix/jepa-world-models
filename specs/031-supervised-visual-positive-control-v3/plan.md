# Implementation Plan: Supervised Visual Positive-Control Study v3

**Branch**: `codex/supervised-visual-positive-control-v3` | **Date**: 2026-10-16 | **Spec**: [spec.md](spec.md)

**Input**: `specs/031-supervised-visual-positive-control-v3/spec.md`

## Summary

Audit the supervised visual dynamics failure in PR #6. Add one narrow v3 validation runner with matched data and model seeds, five baseline roles, one target-scale ablation, and one action-shuffling control. Freeze the method definitions and gates in the v3 preregistration before fitting. Allow one conditional test/rollout stage after the gate. Keep JEPA planning out of scope.

## Technical Context

**Language/Version**: Python 3.10–3.13; experiment environment uses Python 3.10.12

**Primary Dependencies**: Existing PyTorch, Gymnasium, NumPy, Pillow, Matplotlib, and scikit-learn dependencies; no new package

**Storage**: Compact CSV/JSON evidence and figures in the repository; transient frames and checkpoints under `E:\codex-research\jepa-world-models-tutorial\visual-jepa-supervised-positive-control-v3`

**Testing**: Existing unittest suite, Ruff, v3 evidence validation, deterministic figure regeneration, and bounded CUDA validation

**Target Platform**: Local Linux/WSL environment with NVIDIA RTX 5070 and CUDA 12.8; CPU is sufficient for tests and figures

**Project Type**: Research tutorial package with command-line experiment scripts

**Performance Goals**: Complete no more than 15 neural fits of 35 epochs within 60 GPU-minutes

**Constraints**: Peak allocated GPU memory ≤2 GiB; transient output ≤5 GiB on E:; no test data before validation pass; no planning; no changes to historical notebooks or evidence

**Scale/Scope**: Three paired training/validation data and model seeds; 20 shared validation contexts per data seed; five supervised visual method arms/controls

## Constitution Check

| Principle | Plan check |
|---|---|
| Teach from evidence | Claims link to PR #6 code, retained v2 files, or v3 raw records. No new paper claim is made. |
| Label teaching examples honestly | Call this a synthetic MountainCar positive-control study, not a paper reproduction. Label the residual arm as hybrid. |
| Make the learning path usable | Record environment, seeds, data path, commands, expected files, and device limits. Tests and figures remain CPU-runnable. |
| Measure before making performance claims | Lock splits, baselines, thresholds, seed pairs, selection, compute, and stop rules before fitting. Preserve failures. |
| Keep changes small and reviewable | Add one versioned study and reuse the current renderer, encoder, and dependencies. Do not modify Notebook 22/23 or v1/v2 evidence. |

**Gate status**: Pass. The study needs one runner and one figure generator. It adds no dependency or external interface.

## Design Decisions

1. Reuse the existing renderer, two-frame encoding path, `VisualEncoder`, and exact MountainCar environment transition.
2. Keep the v2 absolute-state model unchanged as the primary supervised baseline.
3. Change only training-target scaling in one paired ablation. Use train-only coordinate means and standard deviations.
4. Change only target parameterization for the residual arm. Use the fixed handcrafted state estimate as the skip value and label the arm as hybrid.
5. Shuffle only training action labels in the negative control. Keep validation and evaluation actions correct.
6. Generate test-seed data only from the separately gated held-out command. Do not add or run a planner.
7. Keep full frames and checkpoints on E:. Commit compact source, per-run records, figures, and hashes only.

## Project Structure

```text
specs/031-supervised-visual-positive-control-v3/
  spec.md, plan.md, research.md, data-model.md, quickstart.md, tasks.md
docs/reproductions/protocols/visual-jepa-supervised-positive-control-v3.md
docs/reproductions/visual-jepa-supervised-positive-control-v3-investigation.md
src/jepa_tutorial/supervised_visual_v3.py
scripts/run_visual_jepa_supervised_positive_control_v3.py
scripts/analyze_visual_jepa_supervised_positive_control_v3.py
scripts/validate_visual_jepa_supervised_positive_control_v3.py
tests/test_visual_jepa_supervised_positive_control_v3.py
docs/evidence/visual-jepa-supervised-positive-control-v3/
docs/portfolio/figures/visual-jepa-supervised-positive-control-v3/
```

**Structure Decision**: Keep model/metric helpers in the current package and orchestration in scripts. Store only compact records and figures in Git. Store frames and checkpoints on E:.

## Complexity Tracking

No constitution violations.
