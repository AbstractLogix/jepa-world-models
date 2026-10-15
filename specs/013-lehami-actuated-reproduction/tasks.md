---
description: "Tasks for the scoped LeHami actuated-pendulum reproduction"
---

# Tasks: LeHami Actuated-Pendulum Velocity Reproduction

**Input**: Design documents in `specs/013-lehami-actuated-reproduction/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Verification**: Complete the user-selected seed-0 run for both models, verify the dataset hash and split, evaluate the anchor 25-NN test metric, and label the report partial with seeds 1 and 2 listed as pending. The paper-level comparison still requires all six runs.

## Phase 1: Source and environment setup

- [x] T001 Confirm the paper's Table 1 claim, official LeHami revision, selected configs, split manifest, and LeWorldModel pin in `specs/013-lehami-actuated-reproduction/research.md`.
- [x] T002 Install the derived official training environment in `/tmp/lehami-runtime-env`, clone pinned LeWorldModel under `/tmp/codex-lehami-20261014/lehami/models/third_party/`, and verify CUDA without changing project dependencies.

## Phase 2: User Story 1 - Compare a paper result (Priority: P1)

**Independent Test**: All six model/seed runs use the official 35-epoch configs and matching generated dataset. The official 25-NN K=0 probe produces one held-out test MAE per run, and the report shows each value and the three-seed summary.

- [x] T003 Generate the official dataset in separate `/tmp/lehami-a0q-20261014-home` and `/tmp/lehami-b0s-20261014-home` directories; verify each HDF5 hash against `lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json`.
- [x] T004 Run one epoch of the official training entry point for `a0q` and `b0s` with seed 0 in separate temporary homes, then check only the K=0 25-NN probe path using `scripts/reproductions/lehami_k0_probe.py`; keep smoke output separate from full-run evidence in `/tmp/codex-lehami-20261014`.
- [x] T005 Run the official `a0q` and `b0s` configurations for seed 0 with no training overrides; preserve both checkpoints, logs, and wall time under `/tmp/lehami-*`.
- [x] T006 Run `scripts/reproductions/lehami_k0_probe.py` for the two seed-0 checkpoints; preserve each `mae_omega` value and probe output under `/tmp/codex-lehami-20261014/lehami/experiments/results/`.
- [x] T007 Record the paper values, both local seed-0 values, missing seeds 1 and 2, source revisions, data hash, split, environment, compute, deviations, and limits in `docs/reproductions/lehami-actuated-velocity.md`.

## Phase 3: User Story 2 - Explain evidence and limits (Priority: P2)

**Independent Test**: The roadmap, reproduction matrix, coverage map, and project task record identify the same target, status, and limits; LeWAM remains pending.

- [x] T008 Update `docs/ROADMAP.md`, `docs/REPRODUCTION_MATRIX.md`, `docs/CURRICULUM_COVERAGE.md`, `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` to reflect measured smoke evidence and its limits.
- [x] T009 Run Spec Kit analysis and convergence for `specs/013-lehami-actuated-reproduction/`; keep any appended work outside the user-selected seed-0 scope open and do not start seeds 1 or 2.

## Dependencies and Execution Order

- T001 is complete. T002 must pass before data generation or model runs.
- T003 verifies the data before training. T004 is a pipeline check only. T005 runs the user-selected full seed-0 pair; T006 depends on those two checkpoints.
- T007 follows T006. T008 and T009 follow the recorded result. The report remains partial until all six required runs finish and the three-seed aggregate is available.
- The user selected one full seed per model for this pass. Do not start seeds 1 or 2; keep them as explicit follow-up work.

## Implementation Strategy

1. Verify the isolated environment and exact data before training.
2. Run the one-epoch smoke check, then estimate the full run time.
3. Run the user-selected seed-0 pair and preserve outputs. The other four paper-configured runs remain pending.
4. Write the partial report and align project records with the measured result and its limits.
5. Analyze and converge the feature. Keep LeWAM Push-T evaluation pending.

## Phase 4: Convergence

- [ ] T010 Run seeds 1 and 2 for both `a0q` and `b0s`, evaluate each with the K=0 25-NN probe, and report all six per-seed values plus the three-seed mean and standard deviation per FR-002, FR-003, and SC-002 (partial).
