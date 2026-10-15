# Tasks: Learned Visual JEPA Decision-Usefulness Lesson

**Input**: Design documents in this directory.
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`.

## Phase 1: Setup

**Purpose**: Establish the lesson and record layout.

- [x] T001 Create the lesson module, notebook, and evidence paths listed in `specs/028-visual-jepa-decisions/plan.md`.
- [x] T002 Record disjoint development (1000/2021), smoke (9000/9021), pre-gate diagnostic (1100–1104 / 2121, 2127, 2139, 2153, 2163), and replacement final training seeds (5100–5104 / 7221, 7227, 7239, 7253, 7263), plus the intervention/evaluation seeds and CEM budget. Exclude development, smoke, and pre-gate diagnostic seeds from the final comparison.

## Phase 2: Foundational

**Purpose**: Build and check the small model primitives used by both prediction and planning.

- [x] T003 Add CPU unit checks for model output shapes, finite losses, embedding-spread diagnostics, and device fallback in `tests/test_action_conditioned_visual.py`.
- [x] T004 Implement the compact visual encoder, action-conditioned predictor, image decoder baseline, state-space model, and checked CUDA/CPU selection in `src/jepa_tutorial/action_conditioned_visual.py`.
- [x] T005 Implement the shared transition collector and seed-specific train/validation split in `notebooks/22_visual_jepa_decision_usefulness.ipynb`.

## Phase 3: User Story 1 - Inspect learned visual predictions (Priority: P1)

**Goal**: Train the visual latent and JEPA models and measure their response to held-out action changes.

**Independent Test**: A fresh kernel produces finite records for at least 60 held-out interventions, keeps all candidate actions paired by start, and reports model-specific prediction measures.

- [ ] T006 After a matching validation-only pilot passes, train the privileged state model, image-reconstructing latent model, and JEPA on the five replacement seed groups; report the predeclared collapse screen for each run. Pre-gate diagnostic rows are excluded.
- [ ] T007 After the pilot gate passes, record prediction loss, SIGReg loss, embedding spread, collapse state, device, versions, wall time, and peak GPU memory for final model runs in the notebook and `docs/evidence/visual-jepa-model-runs.csv`.
- [x] T008 Apply the locked validation-only gate. The latest protocol-matched pilot failed; CUDA pilot mode collected zero final intervention starts and zero final episode seeds, and full mode skips final training/evaluation when the matching pilot does not pass. Keep the comparison partial. The evaluation branch remains gated.
- [x] T009 Add a focused record validator for method, seed, start, action, status, and finite metric fields in `tests/test_action_conditioned_visual.py`.

## Phase 4: User Story 2 - Compare decisions on paired starts (Priority: P1)

**Goal**: Compare model-based action selection with oracle, privileged, direct, and policy-guided conditions.

**Independent Test**: All completed methods use the same held-out starts and stated episode and planner limits; each learned model seed has 20 episode records.

- [x] T010 Implement oracle dynamics and label oracle and learned state-space methods as privileged in `notebooks/22_visual_jepa_decision_usefulness.ipynb`.
- [x] T011 Implement uniform CEM and scripted-momentum-prior CEM with the shared horizon, candidate, elite, update, and replanning limits in `notebooks/22_visual_jepa_decision_usefulness.ipynb`.
- [ ] T012 Evaluate the direct visual controller and each model/planner condition on the same 20 paired starts, with 20 episodes for every planned learned-model seed; preserve failed and incomplete rows in `docs/evidence/visual-jepa-episodes.csv`.
- [ ] T013 Report per-seed outcomes, evaluation-case pairing, success, return, prediction measures, compute calls, elapsed time, missing runs, and claim limits in `docs/reproductions/visual-jepa-decision-usefulness.md`.

## Phase 5: User Story 3 - Repeat the local study (Priority: P2)

**Goal**: Make the notebook usable from a clean CPU setup and an optional supported CUDA setup.

**Independent Test**: The quickstart launches the notebook with the locked environment, selects CPU or validated CUDA, and writes the declared records without external assets.

- [x] T014 Complete the learner-facing notebook sections, figures, method labels, and check-your-understanding prompts in `notebooks/22_visual_jepa_decision_usefulness.ipynb`.
- [x] T015 Verify all notebook code cells in a fresh CPU smoke process and pilot mode in the prepared E:-hosted CUDA environment; record the devices and run levels in `docs/reproductions/visual-jepa-decision-usefulness.md`. Neither run opened final cases.
- [x] T016 Validate that CSV row counts, pairing, method labels, model seeds, and failure statuses agree with `specs/028-visual-jepa-decisions/data-model.md`.

## Phase 6: Polish and project alignment

**Purpose**: Keep public project documentation and the project vision in sync with measured evidence.

- [x] T017 Update `README.md`, `docs/CURRICULUM_COVERAGE.md`, `docs/ROADMAP.md`, and `docs/REPRODUCTION_MATRIX.md` with the lesson, primary sources, result level, resource needs, and limits.
- [x] T018 Update `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` to trace the completed or open work without claiming an incomplete comparison.
- [x] T019 Run the focused unit, packaging, notebook, evidence, and Spec Kit consistency checks; record each pass, failure, and not-run check in the reproduction report.
- [x] T020 Add a validation-only pilot mode with a hash of the training code and model/loss modules; gate final model training, intervention collection, and episode evaluation on the latest matching passing pilot. Verify that the current failed pilot keeps all final cases closed.

## Dependencies and execution order

- T001–T005 establish the model and data foundation.
- User Story 1 tasks T006–T009 depend on the foundation.
- User Story 2 tasks T010–T013 depend on the model and held-out data definitions; T012 depends on T010 and T011.
- User Story 3 tasks T014–T016 depend on the completed lesson flow and evidence schema. T020 establishes the pilot gate before T006–T008 can use replacement final seeds.
- Alignment tasks T017–T019 depend on actual implementation and run status.
- The notebook smoke can run before the full comparison. A smoke run does not satisfy the five-seed comparison threshold.

## Implementation strategy

Complete the unit-checked model primitives first. Then build and smoke-test the notebook on CPU. Run a bounded CUDA pilot, estimate the full five-seed duration, and proceed with the fixed full protocol when it is practical. Keep any interrupted result exploratory or partial; do not lower the seed or episode threshold after seeing outcomes.

## Phase 7: Convergence

- [ ] T021 After the protocol-matched pilot passes and final models are available, run the held-out 20-start, three-action intervention protocol for the oracle and each planned model seed. Validate the paired rows and preserve complete, failed, and missing records with method, prediction space, scoring space, and device in `docs/evidence/visual-jepa-interventions.csv`, per FR-004, SC-001, and US1/AC1 (partial).
