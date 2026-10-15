---
description: "Tasks for the LeHami symplectic dynamics tutorial"
---

# Tasks: LeHami Symplectic Dynamics Tutorial

**Input**: Design documents in `specs/014-lehami-symplectic-tutorial/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`

**Tests**: No separate tests requested. Run the notebook to verify its stated local teaching example.

## Phase 1: Setup

- [x] T001 Confirm the Python standard library is sufficient for the calculation and no dependency change is needed in `specs/014-lehami-symplectic-tutorial/plan.md`.

## Phase 2: User Story 1 - See what a symplectic update preserves (Priority: P1)

**Goal**: Explain the structure with a deterministic, CPU-only calculation.

**Independent Test**: The notebook reports leapfrog determinant and polygon-area errors within `1e-12` for each displayed step size. The explicit-Euler contrast is measured and is not labeled as LeWM.

- [x] T002 [US1] Create the learning goals and harmonic-oscillator leapfrog and Euler calculations in `notebooks/15_lehami_symplectic_dynamics.ipynb`.

## Phase 3: User Story 2 - Interpret the paper and local evidence (Priority: P2)

**Goal**: Connect the structural demo to the cited paper and the current partial local result.

**Independent Test**: The notebook separates the Table 1 three-seed statistics from seed-0 local measurements and states the limits of the K=0 probe.

- [x] T003 [US2] Add the LeHami architecture explanation, paper/local K=0 evidence table, and claim limits to `notebooks/15_lehami_symplectic_dynamics.ipynb`.

## Phase 4: Polish and project alignment

- [x] T004 Execute every notebook code cell on CPU, check numeric tolerances and labels, and record runtime and outputs in `docs/evidence/lehami-symplectic-tutorial.md`.
- [x] T005 Update `README.md`, `CURRICULUM_GUIDE.md`, `docs/CURRICULUM_COVERAGE.md`, `docs/ROADMAP.md`, `specs/001-project-vision/tasks.md`, `specs/001-project-vision/alignment.md`, and `specs/001-project-vision/remaining-work.md` with the new lesson and its partial evidence; keep the four seed runs open.
- [x] T006 Analyze and converge `specs/014-lehami-symplectic-tutorial/` against its implementation, then confirm `specs/001-project-vision/tasks.md` marks T034 complete and T033 open.

## Dependencies and Execution Order

- T001 is complete. T002 precedes T003 because both edit the same notebook.
- T004 follows the notebook content tasks. T005 aligns the project records after the lesson content is final. T006 checks the completed implementation and records remaining work.
- The four full training and probe runs for LeHami seeds 1 and 2 are outside this feature and remain pending.

## Parallel Opportunities

- No tasks are marked parallel because the two story tasks update the same notebook and the alignment task depends on the final lesson scope.

## Implementation Strategy

Complete User Story 1 and User Story 2 in order. Execute the notebook and record evidence. Update the project coverage and roadmap. Analyze and converge the feature before reporting completion.
