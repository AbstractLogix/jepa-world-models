# Implementation Plan: LeHami Actuated-Pendulum Velocity Reproduction

**Branch**: `013-lehami-actuated-reproduction` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

## Summary

Attempt to reproduce one LeHami paper result: the actuated-pendulum anchor 25-NN angular-velocity MAE for LeHami (`a0q`) versus LeWM (`b0s`). Use the official generated dataset, shipped episode split, configurations, evaluation code, and three training seeds. Record a bounded local result and its limits. Do not run the planning study, other model arms, or the full LeWAM evaluation.

## Technical Context

**Language/Version**: Python 3.10 or newer as required by the official repository; local host currently has Python 3.13.14  
**Primary Dependencies**: Official LeHami training dependencies from a derived lock; pinned LeWorldModel source; CUDA-enabled PyTorch installed only in the isolated reproduction environment  
**Storage**: Generated dataset, checkpoints, and raw outputs stay under `/tmp`; commit only the compact evidence report and project documentation  
**Verification**: Complete all six official training runs, preserve official evaluation outputs, and reproduce the anchor 25-NN probe from the same held-out episode split. A one-epoch smoke run is a pipeline check, not paper evidence.  
**Target Platform**: Local Linux host with NVIDIA GPU; current preflight reports an RTX 5070 with 12,227 MiB VRAM and driver 617.42  
**Project Type**: Research reproduction report and curriculum/roadmap update; one focused script calls the official probe helpers without adding project dependencies  
**Performance Goals**: Use the official 35-epoch settings and record wall time for each model/seed run  
**Constraints**: Official LeHami commit `f76f65f8b99afb6b7e3063f573645d3e2f290ede` is Apache-2.0. The official runner pins LeWorldModel at `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The official `uv sync` cannot build the unused `env` extra here because `labmaze` needs Bazel. The isolated runtime lock removes only that extra; versions for retained packages match the official lock. Target scripts do not import Gymnasium, OGBench, MuJoCo, or Box2D. Match the generated dataset SHA-256 to the shipped manifest before training. Do not commit data or checkpoints. Keep LeWAM's archive and 450-episode evaluation pending.

## Constitution Check

- Teach from evidence: pass. Use the paper and first-party code as primary sources; report paper and local values separately.
- Label teaching examples honestly: pass. Call this one selected-result reproduction attempt, not a full-paper reproduction or physical validation.
- Make the learning path usable: pass. The generated pendulum data need no private data or robot hardware; disclose the GPU requirement and optional setup.
- Measure before making performance claims: pass. Fix the models, metric, split, seeds, and configurations before viewing final test results.
- Keep changes small and reviewable: pass. Keep the experiment in the official repository clone under `/tmp`; add one report and update existing research records only.

## Research Decisions

1. Target Table 1's actuated-pendulum anchor result at **K=0**: the test-split angular-velocity MAE from the 25-NN probe. The paper reports `0.419 ± 0.045 rad/s` for LeHami and `2.508 ± 0.002 rad/s` for LeWM across three training seeds.
2. Use the paper's main LeHami arm `a0q` and its per-frame LeWM baseline `b0s`. Both official configurations use 35 epochs and batch size 32. Keep each configuration's remaining official settings unchanged.
3. Use the official actuated-pendulum data generator with seed 42 and the shipped 630/135/135 train/validation/test episode manifest. Verify the generated HDF5 SHA-256 against the manifest before training.
4. Use training seeds 0, 1, and 2, as exposed by the official experiment runner. The paper reports three training seeds but does not list their values in Table 1.
5. Use the official training entry point and evaluation helpers. Drive only the K=0 25-neighbor probe through `scripts/reproductions/lehami_k0_probe.py`; this helper verifies the HDF5 hash and stores metrics, the fitted probe, and held-out predictions. Do not run unrelated K=1, MLP, rollout, or planning probes. Keep the six per-run outputs and aggregate only after all required runs finish.
6. Install dependencies in an isolated clone; do not alter this project's environment. Keep generated datasets, checkpoints, and raw outputs outside Git. A one-epoch smoke run checks setup only.
7. If the hash, hardware, run count, or evaluation does not match the plan, record the deviation and mark the attempt partial. Do not tune the protocol after inspecting held-out results.

## Project Structure

```text
specs/013-lehami-actuated-reproduction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── tasks.md
└── checklists/requirements.md

docs/reproductions/lehami-actuated-velocity.md
scripts/reproductions/lehami_k0_probe.py
docs/ROADMAP.md
docs/REPRODUCTION_MATRIX.md
docs/CURRICULUM_COVERAGE.md
specs/001-project-vision/{alignment.md,remaining-work.md,tasks.md}
```

The official code clone, isolated Python environment, generated data, checkpoints, and raw run outputs stay outside the project checkout under `/tmp/codex-lehami-20261014`, `/tmp/lehami-runtime-env`, and separate `/tmp` data homes. The compact report stores the evidence needed to review the result without committing generated assets.

## Complexity Tracking

No constitution violations. No new abstraction or project dependency is needed.
