# Tasks: Paper-aligned JEPA objectives

**Input**: Design documents from `specs/007-paper-aligned-losses/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`.

## Phase 1: Setup and shared objective code

- [x] T001 Add separately named paper-aligned SIGReg and RDMReg functions with explicit generators, input checks, and the formulas in `specs/007-paper-aligned-losses/research.md` to `src/jepa_tutorial/paper_losses.py`.
- [x] T002 Add deterministic shape, finiteness, target-range, and repeatability checks for the new loss functions to `notebooks/02_lejepa_from_scratch.ipynb` and `notebooks/03_rectified_lpjepa.ipynb`
- [x] T010 Increase the shared frozen-probe iteration limit to 1,000 and remove its no-op `n_jobs` option after a clean run showed probe convergence warnings.

## Phase 2: User Story 1 - Compare LeJEPA losses (Priority: P1)

**Goal**: Compare the current SIGReg sketch with a paper-aligned SIGReg run on identical local data and training conditions.

**Independent Test**: Run notebook 02 from a clean kernel on CPU. Confirm the five planned seeds and per-run linear-probe and collapse metrics are present.

- [x] T003 Update `notebooks/02_lejepa_from_scratch.ipynb` with fixed disjoint CIFAR-10 splits, the declared small model, matched training budgets, current and paper-aligned SIGReg settings, and five seeded runs.
- [x] T004 Run notebook 02 from a clean kernel on CPU; capture versions, device, model size, runtime, every seed result, and failures in a temporary execution output.

## Phase 3: User Story 2 - Compare Rectified LpJEPA losses (Priority: P1)

**Goal**: Compare the existing teaching proxies with paper-aligned RDMReg and measure both sparse structure and task information.

**Independent Test**: Run notebook 03 from a clean kernel on CPU. Confirm the five planned seeds report encoder and projector probes, L1 and L0 sparsity, and failures.

- [x] T005 Update `notebooks/03_rectified_lpjepa.ipynb` with fixed disjoint CIFAR-10 splits, a matched backbone and projector, the current proxy arm, the paper-aligned Rectified Laplace arm, and five seeded runs.
- [x] T006 Run notebook 03 from a clean kernel on CPU; capture versions, device, model size, runtime, every seed result, and failures in a temporary execution output.

## Phase 4: Results, alignment, and convergence

- [x] T007 Write `docs/reproductions/lejepa-rectified-lpjepa.md` with primary sources, exact local settings, all seed results, aggregate metrics, uncertain intervals, failures, and explicit paper-versus-local boundaries.
- [x] T008 Align `docs/ROADMAP.md`, `docs/CURRICULUM_COVERAGE.md`, `docs/REPRODUCTION_MATRIX.md`, `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and project task T014 with the executed evidence.
- [x] T009 Run Spec Kit analysis and convergence for this feature. Resolve artifact gaps, append only verified remaining tasks, then mark T014 complete only when the local report and notebook evidence agree. Analysis and convergence found no remaining gaps; no convergence phase was added.

## Dependencies

- T001 precedes T002 through T006.
- T010 precedes T004 and T006.
- T003 precedes T004. T005 precedes T006.
- T004 and T006 precede T007 and T008.
- T009 follows all implementation, execution, reporting, and alignment work.

## Implementation Strategy

Implement the shared objective functions first. Complete the LeJEPA notebook, then the Rectified LpJEPA notebook. Run each from a clean kernel and keep execution outputs outside the repository. Write the report from observed outputs. Align project documents only after the runs finish. Use Spec Kit analysis and convergence as the final feature check.
