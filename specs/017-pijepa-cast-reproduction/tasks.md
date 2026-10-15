# Tasks: PiJEPA CAST Table 2 Reproduction

**Input**: Design documents in `specs/017-pijepa-cast-reproduction/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Execution gate**: Do not start transfer, training, or evaluation until every task in Phase 2 passes. Unknown data terms, missing official checkpoints, an unqualified host, or an unverified fixed case list keeps the run blocked.

## Phase 1: Setup

**Purpose**: Recheck source identity before future execution.

- [x] T001 Recheck the PiJEPA, CAST, V-JEPA-2, and evaluator revisions and update the access date and findings in `specs/017-pijepa-cast-reproduction/research.md` and `docs/reproductions/pijepa-cast-planning.md`.

## Phase 2: Foundational gates

**Purpose**: Review every prerequisite and stop the reproduction if any gate fails.

- [x] T002 Review authoritative CAST data terms and record their source and scope in `docs/reproductions/pijepa-cast-planning.md`; no dataset terms are published in the current card, so this gate is blocked.
- [x] T003 Review first-party access to the required method checkpoints and encoder; record available revisions, sizes, and license findings in `docs/reproductions/pijepa-cast-planning.md`; the exact trained checkpoints remain unavailable or unverified, so this gate is blocked.
- [x] T004 Review compute, storage, and evidence access against the paper's four-H200 setup and 80.8 GB CAST archives; the current 12 GB RTX 5070 is not qualified, and no durable evidence-storage plan is identified, so this gate is blocked.
- [x] T005 Review and record the four evaluator modes and current settings in `docs/reproductions/pijepa-cast-planning.md`; the paper case list is unpublished and the policy sample-count and initial-standard-deviation settings need clarification, so this gate is blocked.

**Checkpoint**: A checked task records that its preflight review is complete; it does not mean that its gate passed. No data transfer or run begins unless the report shows every gate T002 through T005 as passed. Current gate results are blocked.

## Phase 3: User Story 1 - Compare the policy-guided planning result (Priority: P1)

**Goal**: Evaluate all four methods on one fixed CAST validation set and compare the local ATE XY RMSE with Table 2.

**Independent Test**: A reviewer can inspect the pinned run manifest, all per-case outputs, run counts, and aggregate values, and can see paper values separately from local values.

- [ ] T006 [US1] Transfer only approved, pinned data and model assets to local storage; record source, revision, terms, size, and content hash in `docs/reproductions/pijepa-cast-planning.md`; do not commit asset files.
- [ ] T007 [US1] Record the exact four-method evaluator options, per-method Octo sample count, initial-prior standard-deviation behavior or deviation, case-list hash, and run configuration in `docs/reproductions/pijepa-cast-planning.md` before training or evaluation.
- [ ] T008 [US1] Run each stochastic training method for the paper's stated seed count, or at least five independent seeds when the paper gives no count; for frozen checkpoints, record the exact source and hash. Preserve checkpoints and logs outside Git and record seed status, hardware, versions, elapsed time, and artifact hashes in `docs/reproductions/pijepa-cast-planning.md`.
- [ ] T009 [US1] Evaluate PiJEPA, Octo-WM, Octo, and default MPPI on the same fixed cases with the pinned official evaluator; preserve raw per-case outputs outside Git and record their paths and hashes in `docs/reproductions/pijepa-cast-planning.md`.
- [ ] T010 [US1] Calculate each method's per-run ATE XY RMSE; report mean, median, and a 95% interval across independent training runs for trained models, or across evaluation cases for frozen checkpoints, labeling case variation separately from training variation. State the interval method and run count, report paired per-case differences where relevant, retain other Table 2 metrics, and record missing-run details in `docs/reproductions/pijepa-cast-planning.md`.

## Phase 4: User Story 2 - Understand reproduction limits (Priority: P2)

**Goal**: Make the final result status and all deviations clear across project documentation.

**Independent Test**: A reviewer can identify which paper conditions were matched, which were not, and why the local result is `partial`, `exploratory`, or `reproduced`.

- [x] T011 [US2] Set the claim status to Not run because the access and resource gates fail and no local evaluation exists; keep the paper metrics separate in `docs/reproductions/pijepa-cast-planning.md`.
- [x] T012 [US2] Align the PiJEPA entries in `README.md`, `docs/ROADMAP.md`, `docs/REPRODUCTION_MATRIX.md`, `docs/CURRICULUM_COVERAGE.md`, `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` with the Not run status.
- [x] T013 [US2] Review links and claim labels in `specs/017-pijepa-cast-reproduction/` and `docs/reproductions/pijepa-cast-planning.md`; confirm that no paper value is presented as a local result and that `.specify/feature.json` remains on its prior feature.

## Dependencies and execution order

- Phase 1 revalidates sources before the access decisions in Phase 2.
- Phase 2 is a hard gate for every User Story 1 run task. If any gate fails, keep the status `Not run` and record the blocker; do not proceed to T006 through T010.
- User Story 2 follows a completed run or a documented stop at a failed gate. It can report a blocked outcome without changing the claim to reproduced.
- There are no safe parallel tasks in the current plan because source, access, and run records must remain consistent and gated in order.

## Implementation strategy

1. Complete T001 through T005 and stop at any failed gate.
2. Only after every gate passes, complete T006 through T010 for the four methods and all planned seeds and cases.
3. Complete T011 through T013 to align the final evidence status across project documents.

The minimum current outcome is a well-sourced blocked preflight. A partial run is not an MVP for the paper-reproduction claim.
