# Tasks: Planning Limits Lesson

**Input**: Design documents from `specs/004-planning-limits/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`

## Phase 1: Setup

**Purpose**: Add the learner-facing notebook.

- [x] T001 Create notebook 10 with lesson sections and metadata in `notebooks/10_planning_limits.ipynb`.

## Phase 2: Foundational transition model

**Purpose**: Verify the local transition before using it in planner rollouts.

- [x] T002 Implement the vectorized continuous MountainCar transition and compare fixed state/action cases with Gymnasium within absolute tolerance `1e-6` in `notebooks/10_planning_limits.ipynb`.

## Phase 3: User Story 1 - Compare planning methods (Priority: P1)

**Goal**: Run a paired, bounded evaluation of open-loop imagination, feedback MPC, momentum-reference subgoal MPC, and a momentum baseline.

**Independent Test**: Run the notebook on CPU. Confirm every mode uses the same 20 initial-state seeds for each of the three goals, and that each horizon and planner mode shows successes, failures, and missing runs.

- [x] T003 [US1] Implement CEM with 64 candidates, 8 elites, 2 iterations, and the configured horizon, plus the documented momentum rule, in `notebooks/10_planning_limits.ipynb`.
- [x] T004 [US1] Evaluate goals 0.0, 0.25, and 0.45 at horizons 5 and 20 over 20 paired seeds; show success rate, Wilson 95% interval, steps to success, and planning calls in `notebooks/10_planning_limits.ipynb`.

## Phase 4: User Story 2 - Understand feedback and subgoal limits (Priority: P2)

**Goal**: Make the differences among pure imagination, feedback MPC, and momentum-reference subgoal MPC visible and explain their limits.

**Independent Test**: Inspect planner code and plots. Confirm pure imagination never reads later observations, MPC replans from each new observation, and subgoal MPC targets a state five steps ahead on the same-seed momentum reference while retaining the same model and CEM budget.

- [x] T005 [US2] Implement pure imagination, feedback MPC, and momentum-reference subgoal MPC with paired starts; add the method comparison plot and label the scripted reference limit in `notebooks/10_planning_limits.ipynb`.

## Phase 5: User Story 3 - Interpret evidence honestly (Priority: P3)

**Goal**: Keep the paper's results distinct from local measurements and record enough detail to repeat the local run.

**Independent Test**: Read the paper and local result sections. Confirm the paper section names its Meta-World task and paired-episode conditions, and the local section names software, hardware, seeds, model, planner settings, and evidence limits.

- [x] T006 [US3] Add the paper-reported values with primary-source links and separate them from labelled MountainCar measurements in `notebooks/10_planning_limits.ipynb`.
- [x] T007 [US3] Record local run conditions, results, and limits in `docs/reproductions/planning-limits.md`; update the planning row in `docs/REPRODUCTION_MATRIX.md` and lesson/gap rows in `docs/CURRICULUM_COVERAGE.md` and `docs/ROADMAP.md`.

## Phase 6: Polish and cross-cutting checks

**Purpose**: Confirm the notebook is runnable and project tracking is aligned.

- [x] T008 Run the notebook from top to bottom on CPU using `specs/004-planning-limits/quickstart.md`; record the outcome in `docs/reproductions/planning-limits.md` and update the project vision task `T011` in `specs/001-project-vision/tasks.md` only after the evidence is complete.

## Dependencies and Execution Order

- T001 precedes all notebook implementation.
- T002 precedes planner implementation because every predicted transition uses the local model.
- T003 and T004 deliver User Story 1. T005 builds on the shared planner and completes User Story 2.
- T006 and T007 complete User Story 3; T007 depends on measured output from T004/T005.
- T008 depends on notebook and report completion. Spec Kit analysis and convergence follow implementation as a final alignment check.
- No tasks are parallelizable: the notebook is one file, and the documentation report depends on local results.

## Implementation Strategy

Deliver the runnable paired comparison first. Add explanations and source context next. Run the complete notebook once, record evidence, update the curriculum and roadmap, then analyze and converge the Spec Kit artifacts.
