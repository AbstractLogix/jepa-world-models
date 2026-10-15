# Tasks: Continuous Video-Derived Latent Actions

**Input**: Design documents in `specs/006-video-latent-actions/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Tests**: Run notebook 12 on CPU because the feature includes measured comparison and transfer results. Do not add a separate test framework.

## Phase 1: Setup

**Purpose**: Create the runnable lesson scaffold and deterministic generated data.

- [x] T001 Create `notebooks/12_video_latent_actions.ipynb` with lesson goals, paper context, sources, resource needs, and explicit analogue limits.
- [x] T002 Generate small deterministic grayscale clips with train, held-out, new-shape, and new-camera cases. Extract and validate frame moments in `notebooks/12_video_latent_actions.ipynb`.

## Phase 2: User Story 1 - Compare continuous and discrete latent actions (Priority: P1)

**Goal**: Train and compare both methods on identical frame pairs.

**Independent Test**: Confirm that neither model receives generator action labels and both produce finite predictions for the same held-out clips.

- [x] T003 Implement the bounded continuous inverse dynamics model, discrete eight-code model, and shared forward predictor in `notebooks/12_video_latent_actions.ipynb`.
- [x] T004 Train both methods with five fixed seeds and report every run, same-view error, aggregate statistics, continuous-action range, and discrete-code use in `notebooks/12_video_latent_actions.ipynb`.

## Phase 3: User Story 2 - Inspect camera and object transfer (Priority: P2)

**Goal**: Measure what changes when a source action is used with a new shape or camera.

**Independent Test**: Apply matched source actions to held-out shape and camera cases; report separate target prediction errors.

- [x] T005 Add same-view, new-shape, and new-camera transfer evaluation to `notebooks/12_video_latent_actions.ipynb`; state that all clips are generated and the frame features are hand-built.

## Phase 4: User Story 3 - Separate tutorial and paper evidence (Priority: P3)

**Goal**: Record the method, local measurements, and remaining paper-reproduction gates.

**Independent Test**: Confirm that the run report separates the paper setup from the local synthetic analogue and retains every seed and failed run.

- [x] T006 Run notebook 12 from top to bottom on CPU and record versions, settings, all seed results, transfer metrics, limits, and failures in `docs/reproductions/latent-actions.md`.
- [x] T007 Review the paper's official code, checkpoint, and data links; add its selected finding and access gates to `docs/REPRODUCTION_MATRIX.md` without calling the local notebook a reproduction.
- [x] T008 Update `docs/CURRICULUM_COVERAGE.md`, `docs/ROADMAP.md`, and project task T013 in `specs/001-project-vision/tasks.md` only after the notebook and report are complete.

## Phase 5: Run and alignment checks

**Purpose**: Verify the lesson and reconcile the feature artifacts.

- [x] T009 Parse notebook JSON and code cells, run Spec Kit analysis and convergence, fix artifact gaps, and append only actual remaining work to this task list. Analysis and convergence found no remaining feature work.

## Dependencies and Execution Order

- T001 precedes notebook implementation.
- T002 supplies the data for T003 through T005.
- T003 precedes training and metrics in T004. T005 uses the trained methods and matched transfer clips.
- T006 depends on T001 through T005. T007 and T008 depend on the paper review and completed local run report. T009 follows implementation and verification.
- No tasks are parallelizable because code and evidence depend on the generated data and model runs.

## Implementation Strategy

Build the smallest deterministic synthetic clip set first. Train both methods on the same split. Run the full notebook on CPU before writing results or changing project tracking. Keep the paper-reproduction gates open unless a first-party setup is confirmed and the paper experiment is run.
