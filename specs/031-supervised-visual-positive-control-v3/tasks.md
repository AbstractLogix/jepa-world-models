# Tasks: Supervised Visual Positive-Control Study v3

**Input**: Design documents from `specs/031-supervised-visual-positive-control-v3/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Tests**: Requested. Write contract and gate tests before implementation.

**Organization**: Tasks are grouped by the three user stories in `spec.md`.

## Format: `[ID] [P?] [Story] Description`

- **[P]** marks independent work in different files.
- **[Story]** maps a task to the user story in `spec.md`.
- Each task names its primary file path.

## Phase 1: Setup

**Purpose**: Freeze the scope and the methods before implementation.

- [x] T001 Create the v3 protocol and pre-training root-cause plan in `docs/reproductions/protocols/visual-jepa-supervised-positive-control-v3.md`.
- [x] T002 Record v2 code, target scales, and competing causes in `specs/031-supervised-visual-positive-control-v3/research.md`.

## Phase 2: Foundational

**Purpose**: Define deterministic data, model, diagnostics, and fail-closed contracts.

- [x] T003 Add split, seed, action-triplet, finite-value, and gate contract tests in `tests/test_visual_jepa_supervised_positive_control_v3.py`.
- [x] T004 Implement the current-state estimator, absolute dynamics, target-scaled dynamics, residual dynamics, shuffled-action control, and diagnostics in `src/jepa_tutorial/supervised_visual_v3.py`.
- [x] T005 Implement protocol-lock verification, deterministic seed handling, and a validation-only data path in `scripts/run_visual_jepa_supervised_positive_control_v3.py`.

## Phase 3: User Story 1 - Explain the supervised visual failure (Priority: P1)

**Goal**: Recompute matched controls and report measured explanations without treating unresolved causes as facts.

**Independent Test**: Check all run metrics against raw per-seed/per-start records and confirm no historical v1/v2 file changed.

### Tests for User Story 1

- [x] T006 [US1] Test per-coordinate metrics, gradients, action sensitivity, clipping, split isolation, and shuffled-action behavior in `tests/test_visual_jepa_supervised_positive_control_v3.py`.

### Implementation for User Story 1

- [x] T007 [US1] Implement paired training, handcrafted ridge baseline, exact and handcrafted controls, and per-epoch records in `scripts/run_visual_jepa_supervised_positive_control_v3.py`.
- [ ] T008 [US1] Freeze the implementation/protocol hashes and run the bounded validation fits to E: output `visual-jepa-supervised-positive-control-v3/validation`.

## Phase 4: User Story 2 - Respect the validation gate (Priority: P1)

**Goal**: Open independent data only if a fixed learned visual dynamics family passes and the action-shuffle control fails its action-fidelity gate.

**Independent Test**: Force a failed gate in a temporary fixture and confirm the held-out runner does not create reserved test data or output files.

### Tests for User Story 2

- [x] T009 [US2] Test gate selection, held-out isolation, incomplete-run rejection, missing scheduled starts, early-stop arm records, and evidence integrity in `tests/test_visual_jepa_supervised_positive_control_v3.py`.

### Implementation for User Story 2

- [x] T010 [US2] Implement held-out action triplets and 1/5/10/20/60-step predictive rollouts guarded by the passing validation summary in `scripts/run_visual_jepa_supervised_positive_control_v3.py`.

## Phase 5: User Story 3 - Verify and communicate evidence (Priority: P2)

**Goal**: Retain raw validation records and regenerate accessible figures and a clear technical report.

**Independent Test**: Regenerate SVG/PNG twice from the retained CSVs and validate exact hashes and run completeness.

### Tests for User Story 3

- [x] T011 [US3] Test evidence completeness, manifest hashes, and deterministic figure regeneration in `tests/test_visual_jepa_supervised_positive_control_v3.py`.

### Implementation for User Story 3

- [x] T012 [US3] Generate action-response, per-coordinate validation, and learning/gradient figures from retained CSV records in `scripts/analyze_visual_jepa_supervised_positive_control_v3.py`.
- [ ] T013 [US3] Write the measured v3 report and update the evaluation protocol, reproduction matrix, roadmap, and portfolio case study in `docs/reproductions/visual-jepa-supervised-positive-control-v3-investigation.md` and `docs/`.
- [ ] T014 [US3] Validate public evidence hashes, unchanged historical evidence, and the v3 quickstart outputs in `scripts/validate_visual_jepa_supervised_positive_control_v3.py`.

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Run the requested checks and prepare a focused draft PR.

- [ ] T015 Run focused unit tests, Ruff, package build, evidence validation, and CPU smoke checks; record each result in `docs/reproductions/visual-jepa-supervised-positive-control-v3-investigation.md`.
- [ ] T016 Open a draft PR based on PR #6 and attach it; include the exact gate outcome and limits from `docs/reproductions/visual-jepa-supervised-positive-control-v3-investigation.md` in the PR description.

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1** is complete and freezes the intended experiment.
- **Phase 2** defines and tests interfaces before implementation.
- **Phase 3** depends on Phase 2; all study methods share the same data and contexts.
- **Phase 4** is available only when the Phase 3 validation summary passes every frozen gate.
- **Phase 5** depends on complete evidence from every stage that opened.
- **Phase 6** depends on implementation and the final report.

### User Story Dependencies

- **US1 (P1)**: Depends on the foundational model and metric contracts.
- **US2 (P1)**: Depends on the completed US1 validation and action-shuffle sanity check.
- **US3 (P2)**: Uses the complete validation evidence and any gate-authorized held-out evidence.

### Parallel Opportunities

- T003 can be written before T004 and T005; all model work waits for the test contracts.
- Figure rendering and evidence validation use separate scripts, but both wait for the retained CSV schema.
- Training methods are paired within one runner and are not split into concurrent jobs; this keeps seed, data, and compute records consistent.

## Implementation Strategy

1. Complete the frozen protocol and verify its scope.
2. Write tests for data, action, and gate contracts before model code.
3. Implement the validation path and run only the preregistered seed pairs.
4. If validation fails, stop before test data, rollouts, or planning.
5. If validation passes, run only the preregistered held-out predictive stage; never run a planner in v3.
6. Generate all figures from retained records, validate hashes, and prepare a draft PR without merging.
