# Tasks: Action-Conditioned Video Planning Lesson

**Input**: Design documents from `specs/016-action-video-planning/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Tests**: No separate tests requested. Execute the notebook as the local evidence run.

## Phase 1: Setup

**Purpose**: Confirm the lesson fits the current project environment and no new dependency is needed.

- [x] T001 Check the local CPU environment for the existing NumPy, Gymnasium, Pillow, scikit-learn, and Matplotlib dependencies; keep new dependencies out of `pyproject.toml`.
- [x] T002 Confirm the split, planner budget, success rule, and claim limits in `specs/016-action-video-planning/plan.md` match the project evaluation protocol; label the single-fit result exploratory.

## Phase 2: User Story 1 - Learn planning from visual observations (Priority: P1)

**Goal**: Teach a learner to derive features from rendered frames, fit an action-conditioned transition model, and measure held-out prediction and intervention quality.

**Independent Test**: Execute `notebooks/16_action_conditioned_video_planning.ipynb` through its prediction and intervention sections. The model and planner must use visual features only, and test seeds must not overlap training seeds.

- [x] T003 [US1] Add the MountainCar frame renderer, frame-history feature extractor, separated random-action data collection, and fitted polynomial transition model to `notebooks/16_action_conditioned_video_planning.ipynb`.
- [x] T004 [US1] Add encoder calibration, held-out next-step errors, and action-intervention records to `notebooks/16_action_conditioned_video_planning.ipynb`.

## Phase 3: User Story 2 - Compare a policy-guided planning prior (Priority: P2)

**Goal**: Compare uniform-start CEM, controller-prior-start CEM, and their direct controller on the same held-out starts and budget.

**Independent Test**: Run the closed-loop cells and confirm each method uses the same evaluation seeds, success rule, episode cap, CEM candidate count, horizon, and iterations. The CSV must retain every episode status.

- [x] T005 [US2] Add equal-budget CEM, direct visual momentum control, paired closed-loop evaluation, and an illustrative frame-based episode replay to `notebooks/16_action_conditioned_video_planning.ipynb`.
- [x] T006 [US2] Execute the notebook on CPU; record exact versions, settings, raw episode and intervention rows in `docs/evidence/`, and limitations in `docs/reproductions/action-video-planning.md`.

## Phase 4: Alignment and Convergence

**Purpose**: Keep public project guidance and Spec Kit artifacts consistent with the actual notebook and run.

- [x] T007 Update `README.md`, `CURRICULUM_GUIDE.md`, `docs/ROADMAP.md`, `docs/CURRICULUM_COVERAGE.md`, `docs/REPRODUCTION_MATRIX.md`, and project vision alignment/remaining-work files.
- [x] T008 Analyze specification, plan, and tasks for consistency; fix any finding in the feature artifacts.
- [x] T009 Converge `tasks.md` against the completed code, docs, and evidence; retain any remaining paper-reproduction work as open.

## Dependencies & Execution Order

- T001–T002 establish the local dependencies and evaluation boundary.
- T003–T004 deliver the P1 visual prediction flow.
- T005–T006 deliver the P2 planning comparison and evidence.
- T007–T009 align and converge the completed feature.

## Implementation Strategy

1. Deliver the smallest P1 slice: rendered observations, visual features, action-conditioned model, and held-out prediction evidence.
2. Add the policy-prior comparison and direct controller with matched starts and planner budgets.
3. Execute the notebook, preserve run evidence, and update curriculum records.
4. Analyze and converge; keep paper-scale V-JEPA 2-AC and PiJEPA reproductions as separate open work.
