# Tasks: Object-centric LeJEPA Lesson

**Input**: `spec.md`, `plan.md`, and `research.md` in this feature directory
**Verification**: Clean-kernel notebook execution, five runs per method, finite metrics, and consistent report/roadmap/matrix entries.

## Phase 1: Specification and research

- [x] T001 Review the current v2 paper, original LeJEPA code, Oxford-IIIT Pet source, masks, and license in `research.md`.
- [x] T002 Specify the user outcome, reduced study boundary, and evidence requirements in `spec.md` and `plan.md`.

## Phase 2: Notebook and local study

- [x] T003 Add `notebooks/13_object_centric_lejepa.ipynb` with the two matched arms, source trimaps, fixed split, and balanced linear probe.
- [x] T004 Run five seeds per arm, preserve failures, and record actual environment and runtime in the notebook.
- [x] T005 Record source findings, local study settings, initial failed attempts, final metrics, and reproduction limits in `docs/reproductions/object-centric-lejepa.md`.

## Phase 3: Project alignment and convergence

- [x] T006 Update `docs/ROADMAP.md`, `docs/CURRICULUM_COVERAGE.md`, and `docs/REPRODUCTION_MATRIX.md` with current v2 mask facts and the local study status.
- [x] T007 Update `specs/001-project-vision/tasks.md`, `alignment.md`, and `remaining-work.md` with the lesson and open local-run and paper-reproduction work.
- [x] T008 Check JSON, notebook code cells, sources, task coverage, and feature convergence. Mark the feature complete only if the stated local run evidence exists.

## Phase 4: Convergence

- [x] T009 Record a failed run for every planned method and seed when dataset loading fails, and keep the comparison marked incomplete, per SC-001 and US1/AC3 (partial).
