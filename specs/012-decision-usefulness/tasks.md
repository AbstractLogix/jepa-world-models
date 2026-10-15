---
description: "Tasks for the local decision-usefulness evaluation lesson"
---

# Tasks: Decision-Usefulness Evaluation Lesson

**Input**: Design documents in `specs/012-decision-usefulness/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Verification**: Run the notebook end to end. Confirm disjoint seed splits, at least 60 intervention outcomes, 20 paired episodes per method, and complete compute and missing-run records.

## Phase 1: Lesson setup

- [X] T001 Add the lesson goal, evidence labels, primary sources, fixed evaluation protocol, and existing-dependency constraint to `notebooks/14_decision_usefulness.ipynb`.

## Phase 2: User Story 1 - Inspect action interventions (Priority: P1)

**Independent Test**: The notebook reports at least 60 interventions from 20 held-out states, preserves every action outcome, and calculates prediction error and pairwise action-ranking agreement.

- [X] T002 Generate simulator transitions with separate training and intervention seed groups; fit the state-transition model in `notebooks/14_decision_usefulness.ipynb`.
- [X] T003 Evaluate actions -1, 0, and 1 from at least 20 held-out states; report raw predictions, observed states, prediction error, ranking agreement, and excluded ties in `notebooks/14_decision_usefulness.ipynb`.
- [X] T004 Explain the intervention protocol, data split, metrics, and claim limits in `notebooks/14_decision_usefulness.ipynb`.

## Phase 3: User Story 2 - Compare closed-loop decisions (Priority: P1)

**Independent Test**: The learned-model planner and direct policy run on 20 identical evaluation starts, use the same episode limit and compute cap, and retain all outcomes.

- [X] T005 Implement learned-model MPC and the direct momentum policy in `notebooks/14_decision_usefulness.ipynb`.
- [X] T006 Run 20 paired evaluation episodes per method with a 300-step limit and the same 60-second elapsed-time cap; record per-episode outcomes, elapsed time, and model-call counts in `notebooks/14_decision_usefulness.ipynb`.
- [X] T007 Summarize success, return, paired differences, compute use, and missing runs in `docs/reproductions/decision-usefulness.md`; label the result as an illustrative analogue.

## Phase 4: Align project records and converge

- [X] T008 Update `docs/CURRICULUM_COVERAGE.md`, `docs/ROADMAP.md`, `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` after the notebook results are recorded.
- [X] T009 Run Spec Kit analysis and, after the implementation pass, converge `spec.md`, `plan.md`, and `tasks.md` against the notebook and report.

## Dependencies and Execution Order

- T001 defines the protocol before data collection or final evaluation.
- T002 depends on T001. T003 and T004 depend on the fitted model and fixed intervention set.
- T005 can follow the shared protocol in T001. T006 depends on T005 and the trained model in T002.
- T007 depends on the completed evaluation in T006.
- T008 and T009 follow the recorded report. Do not label the feature complete while any required run is missing or the compute cap is not met.
