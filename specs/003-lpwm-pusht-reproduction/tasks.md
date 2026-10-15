# Tasks: Bounded LpWM PushT Reproduction

**Input**: Design documents from `/specs/003-lpwm-pusht-reproduction/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`

## Setup

- [x] T001 [P] Record the selected claim, source revision, configuration, and pre-run decision rule in `specs/003-lpwm-pusht-reproduction/research.md`
- [x] T002 [P] Verify the filtered two-cell author script in the pinned LpWM checkout
- [x] T003 Verify the downloaded OSF archive size and SHA-256, then extract it under `/tmp/lpwm-data`
- [x] T004 Prepare the isolated Python 3.9 CPU environment and complete the training/planning import and device preflight

## User Story 1 - Paired model training (Priority: P1)

**Goal**: Train sparse LpWM and dense LeWM MLP∘LTV models at D=768 with the same dataset and stated budget.

**Independent Check**: Both expected run directories contain their configuration, logs, and checkpoint, or their failure evidence is recorded.

- [x] T005 [US1] Attempt the selected author training workflow with data and checkpoints under `/tmp`; preserve the partial run and failure evidence
- [x] T006 [US1] Record the attempted sparse arm's configuration, exit status, elapsed time, device, versions, memory, and checkpoint status; state that dense training did not start

## User Story 2 - Planner evaluation (Priority: P2)

**Goal**: Compare complete checkpoints with the same upstream CEM-MPC settings for 50 episodes at local seeds 99, 100, and 101.

**Independent Check**: Each planned run has a seed, episode count, success count, success rate, and completion status.

- [x] T007 [US2] Record that the official seed-99 planner did not run because no training checkpoint was produced
- [x] T008 [US2] Record that planner seeds 100 and 101 did not run because no training checkpoint was produced
- [x] T009 [US2] Confirm that planner scores are unavailable and that every missing run is explicit in the report

## User Story 3 - Evidence report (Priority: P3)

**Goal**: State what the local attempt supports and where the evidence stops.

**Independent Check**: Paper values and local measurements are separate, and the report identifies missing runs, terms, compute, deviations, and partial scope.

- [x] T010 [US3] Write `docs/reproductions/lpwm-pusht.md` from observed evidence using the feature data model
- [x] T011 [US3] Update `docs/REPRODUCTION_MATRIX.md` with the observed run status and evidence report
- [x] T012 [US3] Reconcile the feature spec, task states, project vision task, and reproduction matrix; run Spec Kit analyze and converge

## Dependencies

- T003 and T004 must pass before T005.
- T005 and T006 must complete before T007 and T008.
- T007 and T008 must complete before T009. T010 follows the run evidence; explicit not-run status is valid when a required checkpoint is unavailable.
- T010 must precede T011; both precede T012.

If a run cannot proceed, record the exact blocker in the report and leave dependent tasks incomplete or mark them complete only when the failed attempt itself is documented.

T005 records an incomplete attempt, not completed model training. T007–T009 record that planner runs and scores were unavailable because no checkpoint was produced; they do not claim that planner evaluation ran.

## Phase 5: Convergence

- [ ] T013 [US1] Complete both sparse and dense D=768 training runs with the verified PushT archive on a host with a safe resource plan; record any change to the upstream worker setting per FR-003 and SC-003 (partial)
- [ ] T014 [US2] Evaluate each completed checkpoint for 50 trajectories at planner seeds 99, 100, and 101; report per-seed success, mean, standard deviation, and missing runs per FR-005 and SC-004 (partial)
