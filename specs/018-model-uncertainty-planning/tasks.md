# Tasks: Calibrate and Stress-Test a World Model

**Input**: Design documents in `specs/018-model-uncertainty-planning/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Tests**: No separate test suite is requested. Execute Notebook 17 in a fresh CPU kernel and keep the evidence assertions in the notebook.

## Phase 1: Setup

**Purpose**: Use the current project environment without adding dependencies.

- [X] T001 Confirm the locked NumPy, SciPy, scikit-learn, Matplotlib, Gymnasium, and Jupyter versions are available for the notebook; keep `pyproject.toml` and `poetry.lock` unchanged.

## Phase 2: User Story 1 - Measure uncertainty calibration (Priority: P1)

**Goal**: Fit five independent bootstrap ensembles and test calibrated next-state intervals on separate data splits.

**Independent Test**: Execute the calibration sections of `notebooks/17_model_uncertainty_planning.ipynb`; verify disjoint seed groups, calibration-only scale fitting, and separate coverage/width outputs for random-action and policy-shift transitions.

- [X] T002 Add fixed seed groups, capped MountainCarContinuous episode collection, and five independent degree-three ridge bootstrap ensembles to `notebooks/17_model_uncertainty_planning.ipynb`; persist each run's seed, sample count, training time, software versions, and status to `docs/evidence/model-uncertainty-runs.csv`.
- [X] T003 Fit component-wise 90% interval scales on calibration episodes only; measure and save per-run, per-component coverage and width for in-distribution and policy-shift transitions in `notebooks/17_model_uncertainty_planning.ipynb` and `docs/evidence/model-uncertainty-calibration.csv`.

## Phase 3: User Story 2 - Measure model-plan exploitability (Priority: P2)

**Goal**: Compare model-scored plans with simulator outcomes and closed-loop policy results.

**Independent Test**: Execute the planning sections of `notebooks/17_model_uncertainty_planning.ipynb`; confirm each CEM method uses the same budget and paired starts, every top plan has a same-state/same-sequence simulator return, and the direct policy has one outcome per shared start.

- [X] T004 Add mean-return CEM, fixed mean-minus-one-standard-deviation CEM, the direct momentum baseline, and same-sequence simulator evaluation with signed selected-plan gaps to `notebooks/17_model_uncertainty_planning.ipynb`.
- [X] T005 Run both learned planners on 20 paired evaluation seeds for each of five model runs, run the direct policy on the same 20 starts, and save every plan and episode row to `docs/evidence/model-uncertainty-plans.csv` and `docs/evidence/model-uncertainty-episodes.csv`.
- [X] T006 Summarize per-run success, return, selected-plan gap, mean, median, and 95% Student-t intervals across five model runs in `notebooks/17_model_uncertainty_planning.ipynb`.

## Phase 4: Evidence and alignment

**Purpose**: Execute the lesson and align project claims to its measured evidence.

- [X] T007 Execute `notebooks/17_model_uncertainty_planning.ipynb` in a fresh CPU kernel; record versions, runtime, exact settings, missing rows, and claim limits in `docs/reproductions/model-uncertainty-planning.md`.
- [X] T008 Update `README.md`, `CURRICULUM_GUIDE.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/ROADMAP.md`, `docs/CURRICULUM_COVERAGE.md`, `docs/REPRODUCTION_MATRIX.md`, `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` with the local analogue status and open paper-reproduction boundary.
- [X] T009 Analyze and converge `specs/018-model-uncertainty-planning/tasks.md` against the completed notebook, evidence, and alignment documents; retain any incomplete seed or evidence item as open.

## Dependencies and execution order

- T001 must pass before notebook execution.
- T002 blocks T003, T004, and T005 because all use the fitted ensembles and fixed splits.
- T003 calibration data and scale fitting must finish before any policy-shift or final test metrics are calculated.
- T004 defines planning budgets and score behavior before T005 runs final evaluation.
- T006 and T007 summarize completed raw records; T008 aligns docs to the report; T009 checks the implemented feature for remaining gaps.
- No tasks are marked parallel because this single notebook shares its model, seed splits, and evidence records across both stories.

## Implementation strategy

1. Complete setup and the P1 uncertainty-calibration path.
2. Add the P2 planner comparison and exploitability record.
3. Execute all five model runs and 20 paired final starts; keep all failures and missing rows.
4. Align the project documentation, analyze the feature, and converge its remaining tasks.
