---
description: "Tasks for the policy-noise planning tutorial"
---

# Tasks: Policy-Noise Planning Tutorial

**Input**: Design documents in `specs/022-policy-noise-planning-tutorial/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Verification**: Run every notebook code cell in a fresh CPU kernel. Confirm that the two planners have equal candidate budgets and horizons, that all methods share the same paired starts, and that the report separates toy measurements from the paper claim.

## Phase 1: Research and lesson design

- [x] T001 Review the primary LeWAM diffusion-steering paper and project constitution; record verified method details and the local claim boundary in `research.md`.
- [x] T002 Define the user outcome, measurable acceptance criteria, toy entities, and CPU-only validation path in `spec.md`, `plan.md`, `data-model.md`, and `quickstart.md`.

## Phase 2: User Story 1 - Compare action-space and policy-noise planning (Priority: P1)

**Independent Test**: Run every code cell on CPU. Verify a same-task comparison of raw-action sampling, policy-noise sampling, and direct control; equal planner sampling budgets; paired starts; per-seed success and compute; off-support prediction error; and explicit analogy limits.

- [x] T003 Create `notebooks/20_policy_noise_planning.ipynb` with the paper-backed method explanation, the hand-authored toy task and policy, matched planners, direct-control reference, measurements, and claim limits.
- [x] T004 Execute all notebook cells in a fresh CPU kernel and record software, settings, aggregate results, model calls, runtime, and limits in `docs/reproductions/policy-noise-planning.md`. Keep raw per-episode records out of Git.
- [x] T005 Align `README.md`, `CURRICULUM_GUIDE.md`, `docs/CURRICULUM_COVERAGE.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/evidence/evaluation-protocol-audit.md`, `docs/ROADMAP.md`, `docs/REPRODUCTION_MATRIX.md`, `specs/001-project-vision/research.md`, `specs/001-project-vision/tasks.md`, `specs/001-project-vision/alignment.md`, and `specs/001-project-vision/remaining-work.md` with the local lesson, protocol evidence, and separate official reproduction gate.
- [x] T006 Analyze and converge feature 022 against its specification, plan, tasks, implementation, and local evidence. No remaining feature work was found; the official LeWAM paper evaluation remains not run.

## Dependencies

- T001 and T002 define source-backed claims and the toy data model before implementation.
- T003 precedes T004.
- T004 precedes T005 and T006.
- T006 is last; any new convergence task must follow the append-only Spec Kit rule.

## Implementation Strategy

Deliver one independently runnable CPU lesson. Do not download assets, add dependencies, run model training, or call the toy outcome a paper reproduction.
