---
description: "Tasks for the official LeWAM Push-T checkpoint evaluation"
---

# Tasks: Released LeWAM Push-T Checkpoint Reproduction

**Input**: Design documents in `specs/010-lewam-pusht-reproduction/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Verification**: Verify artifact pins and hashes, then run the official evaluator for the declared policies and seeds. The raw evaluation itself is the main scientific check.

## Phase 1: Preflight and artifact setup

**Purpose**: Confirm that the full attempt can run safely outside the project source tree.

- [X] T001 Check free storage, Python 3.11 setup, CPU/GPU runtime, and headless MuJoCo support. Record the preflight outcome in `docs/reproductions/lewam-pusht.md`.
- [ ] T002 Create an isolated checkout of `MinghaoFu/lewam` at commit `60732e25ba172b117d0d0a476d3a9dcd5d30c03b` under `$LEWAM_RUN_ROOT/code` and install only its declared environment under `$LEWAM_RUN_ROOT/conda-envs`, with package caches under the same E: or D: run root.
- [ ] T003 Download the model under `$STABLEWM_HOME/checkpoints` and dataset under `$STABLEWM_HOME/datasets` at revisions `d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2` and `63346c94f4dbe62b4744261c6c99d63cbff3f6e6`; verify both SHA-256 hashes and prepare the HDF5 file on the selected E: or D: drive.

## Phase 2: User Story 1 - Re-evaluate the released Push-T checkpoint (Priority: P1)

**Goal**: Run the official Push-T settings for each released-checkpoint evaluation mode.

**Independent Test**: Confirm that each of the three modes has 50 episode outcomes for seeds 42, 0, and 1, with the official 25-step goal offset and 50-action budget.

- [ ] T004 [US1] Run the official reactive policy for seeds 42, 0, and 1; save all 150 episode results under `$STABLEWM_HOME/eval/` on the selected E: or D: drive.
- [ ] T005 [US1] Run the official best-of-32 planner for seeds 42, 0, and 1; save all 150 episode results under `$STABLEWM_HOME/eval/` on the selected E: or D: drive.
- [ ] T006 [US1] Run the official gradient planner for seeds 42, 0, and 1; save all 150 episode results under `$STABLEWM_HOME/eval/` on the selected E: or D: drive.

## Phase 3: User Story 2 - Record provenance and limits (Priority: P2)

**Goal**: Compare local rates with paper values and keep the claim boundary visible.

**Independent Test**: Check the report against the official source pins and raw evaluator files. Confirm it states that the evaluator samples training data.

- [ ] T007 [US2] Summarize each mode's success rate by seed and across seeds; label its mean within/outside the paper's reported one-standard-deviation range; record the paper values, local versions, device, runtimes, hashes, errors, and deviations in `docs/reproductions/lewam-pusht.md`.
- [X] T008 [US2] Align `docs/REPRODUCTION_MATRIX.md`, `docs/ROADMAP.md`, `docs/CURRICULUM_COVERAGE.md`, and `specs/001-project-vision/{alignment.md,remaining-work.md,tasks.md}` with the actual run status.
- [ ] T009 [US2] Analyze the feature artifacts and, after implementation, converge `spec.md`, `plan.md`, and `tasks.md` against the actual evaluator results and report.

## Dependencies and Execution Order

- T001 is complete as a recorded preflight, but it found the environment is not ready: Python 3.11 is absent, MuJoCo and HDF5 support are absent, and headless MuJoCo is unverified. T002 and T003 remain pending until a supported environment is available and the pending transfer scope is reopened.
- T002 depends on a T001 preflight that confirms a usable runtime; the current preflight did not. T003 also depends on the pinned checkout path.
- T004, T005, and T006 depend on the code, checkpoint, and dataset in T002–T003. Run them one at a time to avoid competing for local compute.
- T007 and T008 depend on the finished evaluation or a recorded blocking failure. T008 is complete for the recorded preflight status; T007 remains open because no local episode rates or paper-range comparisons exist.
- T009 follows implementation and documentation alignment.

## Implementation Strategy

1. Verify the official sources, runtime, storage, and data path.
2. Run a small evaluator preflight only to detect setup errors; do not report it as reproduction evidence.
3. Run all 450 episodes under the official settings if the preflight passes.
4. Record each run and update project status. Keep missing or failed modes visible as partial work.
