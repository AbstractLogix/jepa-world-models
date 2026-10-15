---
description: "Tasks for the LeWAM RoboTwin 2.0 readiness preflight"
---

# Tasks: LeWAM RoboTwin 2.0 Preflight

**Input**: Design documents in `specs/027-lewam-robotwin-preflight/`

**Verification**: Check report claims against the cited sources, confirm the host observations, and validate local Markdown links. No model evaluation is in scope.

## Phase 1: Setup

**Purpose**: Pin source facts and the host state used by the preflight.

- [x] T001 Review the official paper, code, model, and simulator sources and record the source revisions and host observations in `specs/027-lewam-robotwin-preflight/research.md`.

## Phase 2: Foundational

**Purpose**: Define the evidence records and report shape.

- [x] T002 Define the candidate, source, readiness gate, and run evidence records in `specs/027-lewam-robotwin-preflight/data-model.md` and the documentation-only implementation in `specs/027-lewam-robotwin-preflight/plan.md`.

## Phase 3: User Story 1 - Decide whether a RoboTwin evaluation can start

**Goal**: Give reviewers a source-pinned, evidence-limited readiness report before any evaluation assets are transferred.

**Independent Test**: The report names the paper claim and reviewed code revision, records the WSL/runtime and license gates, states no evaluation ran, and defines a fixed-case restart path.

- [x] T003 [US1] Create `docs/reproductions/lewam-robotwin-preflight.md` with paper, code, model, simulator, host, terms, and stop-gate evidence.
- [x] T004 [US1] Update `docs/ROADMAP.md`, `docs/REPRODUCTION_MATRIX.md`, `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` with the shared not-run status.
- [x] T005 [US1] Validate local links and review the claim boundaries in `docs/reproductions/lewam-robotwin-preflight.md`, `docs/ROADMAP.md`, and `docs/REPRODUCTION_MATRIX.md`.
