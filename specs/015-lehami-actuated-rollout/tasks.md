---
description: "Tasks for the bounded LeHami seed-0 rollout evaluation"
---

# Tasks: LeHami Seed-0 Actuated Rollout Evaluation

**Input**: Design documents in `specs/015-lehami-actuated-rollout/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`

**Tests**: No separate test suite. Run the official evaluation and validate its raw outputs.

## Phase 1: Setup and preflight

- [x] T001 Verify the official source revision, both seed-0 checkpoint hashes, the dataset hash, the official manifest, the existing CUDA runtime, and unique result names in `specs/015-lehami-actuated-rollout/`.

## Phase 2: User Story 1 - Measure prediction across a rollout (Priority: P1)

**Goal**: Measure held-out K=1-probe angular-velocity error across the official actuated-pendulum rollout for each saved seed-0 model.

**Independent Test**: Each official evaluation exits successfully and writes finite per-step probe metrics through K=76. The report uses the held-out test split and keeps the two one-seed results separate.

- [x] T002 Run the official standard evaluator for the existing `a0q` seed-0 checkpoint at horizon 76 and save raw output under a new `codex_` result name in `/tmp/codex-lehami-20261014/lehami/experiments/results/`.
- [x] T003 Run the official standard evaluator for the existing `b0s` seed-0 checkpoint at horizon 76 and save raw output under a different new `codex_` result name in `/tmp/codex-lehami-20261014/lehami/experiments/results/`.
- [x] T004 Validate the two raw output directories and record source, checkpoint and data hashes, per-step test metrics, runtime, hardware, commands, deviations, and limits in `docs/reproductions/lehami-actuated-velocity.md`.
- [x] T005 Add the seed-0 rollout comparison and clear evidence limits to `notebooks/15_lehami_symplectic_dynamics.ipynb`.

## Phase 3: Project alignment

- [x] T006 Update `docs/CURRICULUM_COVERAGE.md`, `docs/ROADMAP.md`, `specs/001-project-vision/tasks.md`, `specs/001-project-vision/alignment.md`, and `specs/001-project-vision/remaining-work.md` with the new single-seed rollout evidence; keep T033 open.
- [x] T007 Analyze and converge `specs/015-lehami-actuated-rollout/` against the saved outputs, notebook, and project records.

## Dependencies and Execution Order

- T001 must pass before either evaluation starts. T002 and T003 run sequentially because they use the same CUDA device.
- T004 validates both outputs and records the evidence before T005 adds the selected measurements to the notebook.
- T006 follows the final evidence and notebook content. T007 is the final Spec Kit gap check.
- The four additional training runs, planning evaluation, and LeWAM Push-T evaluation remain outside this feature.

## Parallel Opportunities

- None. Both model evaluations use the same GPU and all later work depends on their outputs.

## Implementation Strategy

Complete preflight, run the two existing checkpoints with the pinned evaluator, validate and report the held-out results, update Notebook 15 and project alignment, then analyze and converge this feature.
