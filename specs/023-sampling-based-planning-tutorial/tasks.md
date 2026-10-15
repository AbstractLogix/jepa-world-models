---
description: "Tasks for the sampling-based planning tutorial"
---

# Tasks: Sampling-Based Planning Tutorial

**Input**: Design documents in `specs/023-sampling-based-planning-tutorial/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Verification**: Run every notebook cell in a fresh CPU kernel. Confirm common starts, a shared transition model, equal maximum transition-call budgets per decision, complete planned episode counts, and explicit analogy limits.

## Phase 1: Research and design

- [x] T001 Review the primary paper and author-linked artifacts; define the source-backed claims, toy boundary, entities, and local run in `research.md`, `spec.md`, `plan.md`, `data-model.md`, and `quickstart.md`.

## Phase 2: User Story 1 - Compare tree search and shooting (Priority: P1)

**Independent Test**: Run every notebook code cell in a fresh CPU kernel. Verify RRT, KPIECE, and CEM use the same task, transition function, shared starts, and maximum transition-call budget; check episode counts, metrics, and claim labels in the regenerated report.

- [x] T002 Create `notebooks/21_sampling_based_planning.ipynb` with the toy task, RRT and KPIECE expansion, CEM shooting baseline, matched closed-loop evaluation, useful plots, and explicit source and claim limits.
- [x] T003 Execute the full notebook in a fresh CPU kernel; record run conditions, aggregate and per-group outcomes, missing/error counts, and analogy limits in `docs/reproductions/sampling-based-planning.md`.
- [x] T004 Align `README.md`, `CURRICULUM_GUIDE.md`, `docs/CURRICULUM_COVERAGE.md`, `docs/ROADMAP.md`, and `docs/REPRODUCTION_MATRIX.md` with Notebook 21 and the separately gated paper evaluation.
- [x] T005 Apply `docs/EVALUATION_PROTOCOL.md` to the Notebook 21 report; update `docs/evidence/evaluation-protocol-audit.md`, `specs/001-project-vision/research.md`, `specs/001-project-vision/tasks.md`, `specs/001-project-vision/alignment.md`, and `specs/001-project-vision/remaining-work.md` with observed evidence and remaining gates.
- [x] T006 Analyze and converge this feature after implementation and documentation updates; append only any remaining in-scope work to this file.

## Dependencies

- T001 defines claims and scope before implementation.
- T002 precedes T003.
- T003 precedes T004 and T005.
- T006 is last; any convergence task must follow the append-only Spec Kit rule.

## Implementation strategy

Deliver one CPU notebook and its concise local report. Do not download assets, add dependencies, or label the toy result a paper reproduction.
