---
description: "Tasks for repeatable local runs of the original tutorial lessons"
---

# Tasks: Reproducible Legacy Lesson Evidence

**Input**: Design documents in this feature directory

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Verification**: Execute notebooks 01 and 04–08 in clean CPU kernels. Repeat notebooks 05 and 06 and compare their seeded episode results. Record all results and failures in `docs/evidence/legacy-lessons-local-run.md`.

## Phase 1: Repeatable notebook behavior

### User Story 1 - Run the original lessons locally (Priority: P1)

- [X] T001 Fix the covariance ellipse solver and seed the toy run in `notebooks/01_intuition.ipynb`.
- [X] T002 Pin and record the pretrained model source and fixed seeds, fit on the training split only, report held-out synthetic descriptor alignment/retrieval, and save to an ignored checkpoint path in `notebooks/04_vjepa_plus_v1_jepa.ipynb`.
- [X] T003 Seed the collected trajectories and use the same 25 evaluation seeds for both policies in `notebooks/05_value_guided_planning.ipynb`.
- [X] T004 Use the same 20 evaluation seeds for both planners and seed candidates per episode in `notebooks/06_gradient_based_planning.ipynb`.
- [X] T005 Seed environment action sampling and resets, then label the training-set metric in `notebooks/07_latent_action_world_models.ipynb`.
- [X] T006 Seed collected data, model initialization, and matched evaluation cases in `notebooks/08_physical_planning_ablation.ipynb`.

## Phase 2: Local execution evidence

### User Story 1 - Run the original lessons locally (Priority: P1)

- [X] T007 Execute notebooks 01 and 04 from clean CPU kernels and save the executed copies outside the repository under `/tmp/jepa-local-evidence-20261014/`.
- [X] T008 Execute notebooks 07 and 08 from clean CPU kernels and save the executed copies outside the repository under `/tmp/jepa-local-evidence-20261014/`.

### User Story 2 - Compare planning methods on matched cases (Priority: P1)

- [X] T009 Execute notebooks 05 and 06 twice in separate temporary roots; confirm that each method uses the same episode seeds and that repeated episode returns match in the same CPU environment.

## Phase 3: Evidence and alignment

### User Story 3 - Keep evidence and curriculum records aligned (Priority: P2)

- [X] T010 Record raw results, versions, device, runtime, failures, and claim limits in `docs/evidence/legacy-lessons-local-run.md`.
- [X] T011 Align `docs/CURRICULUM_COVERAGE.md`, `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` with the measured evidence.
- [X] T012 Analyze and converge `specs/009-reproducible-legacy-lessons/spec.md`, `plan.md`, and `tasks.md` against the notebook code and run report. Analysis found full requirement/task coverage; convergence found no remaining work and appended no tasks.

## Dependencies and Execution Order

- T001 through T006 change separate notebooks and can be implemented independently.
- T007 and T008 depend on T001, T002, T005, and T006.
- T009 depends on T003 and T004; it also verifies the shared checkpoint path between notebooks 05 and 06.
- T010 and T011 depend on completed local runs. T012 follows implementation and documentation alignment.

## Implementation Strategy

1. Make seeds, paired evaluation cases, and ignored checkpoint paths explicit in the notebooks.
2. Run the small self-contained lessons and the paired planning lessons in clean CPU kernels.
3. Record all local results and remaining limits. Keep the single-seed ablation exploratory.
4. Align the project coverage records, then run Spec Kit analysis and convergence.
