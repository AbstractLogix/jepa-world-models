---
description: "Implementation tasks for the V-JEPA 2.1 dense-feature tutorial"
---

# Tasks: V-JEPA 2.1 Dense Video Features Tutorial

**Input**: Design documents in `specs/002-vjepa-21-dense-features/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Tests**: User request requires reproducible local evidence. Run the no-download notebook path and an opt-in official-checkpoint CPU path.

## Phase 1: Setup

**Purpose**: Declare the optional dependency needed by the pinned upstream model loader.

- [x] T001 Add the `vjepa21` optional dependencies and synchronize `pyproject.toml` with `poetry.lock`.

## Phase 2: User Story 1 - Inspect dense video features (Priority: P1)

**Goal**: Generate a deterministic local clip and inspect dense features from the named official encoder.

**Independent Test**: The notebook's clip generation and plotting sections run without network access. With the opt-in checkpoint available, the encoder returns finite features with eight temporal groups and a 24×24 patch grid.

- [x] T002 [US1] Add learner outcome, prerequisites, and a deterministic 16-frame 384×384 clip generator to `notebooks/09_vjepa21_dense_features.ipynb`.
- [x] T003 [US1] Add clip visualization and frame/object-position metadata to `notebooks/09_vjepa21_dense_features.ipynb`.
- [x] T004 [US1] Add explicit opt-in loading of the pinned official Hub encoder and checkpoint with strict `ema_encoder` weights to `notebooks/09_vjepa21_dense_features.ipynb`.
- [x] T005 [US1] Apply official preprocessing and display finite features with eight temporal groups and a 24×24 patch grid in `notebooks/09_vjepa21_dense_features.ipynb`.
- [x] T006 [US1] Run the no-download sections and the official checkpoint CPU path; record input, output shape, device, preprocessing, elapsed time, and memory in `docs/evidence/vjepa21-local-run.md`.

## Phase 3: User Story 2 - Compare temporal patch correspondence (Priority: P2)

**Goal**: Measure one known moving-object patch retrieval for V-JEPA features and a named pixel baseline.

**Independent Test**: Both methods report query and known target patches, retrieved patch, cosine similarity, and target distance for the same generated clip; no success threshold is used.

- [x] T007 [US2] Add target-patch mapping, feature cosine retrieval, and mean-RGB patch baseline to `notebooks/09_vjepa21_dense_features.ipynb`.
- [x] T008 [US2] Execute and record both correspondence measurements with run conditions in `docs/evidence/vjepa21-local-run.md`.

## Phase 4: User Story 3 - Run within local resource limits (Priority: P3)

**Goal**: Make resource needs, checkpoint provenance, terms uncertainty, and optional paths clear before a learner starts.

**Independent Test**: The no-download path succeeds without fetching weights; the README and curriculum documents state the official checkpoint, its approximate size, the 384-pixel model input, the CPU path, and the non-reproduction boundary.

- [x] T009 [US3] Add checkpoint provenance, file-size reporting, terms status, device and peak-memory reporting, and explicit download guidance to `notebooks/09_vjepa21_dense_features.ipynb`.
- [x] T010 [US3] Add the lesson to the learner path and align the lesson description in `README.md`, `CURRICULUM_GUIDE.md`, `docs/CURRICULUM_COVERAGE.md`, and `docs/ROADMAP.md`.
- [x] T011 [US3] Validate clean notebook execution without weights and CPU execution with the official checkpoint; report actual limits and results in `docs/evidence/vjepa21-local-run.md`.

## Phase 5: Polish and cross-cutting checks

**Purpose**: Reconcile specifications and task status with the implemented lesson.

- [x] T012 Review `specs/002-vjepa-21-dense-features/spec.md`, `plan.md`, `tasks.md`, `docs/CURRICULUM_COVERAGE.md`, and `docs/ROADMAP.md` for agreement and mark project task T010 complete in `specs/001-project-vision/tasks.md` only after the local checkpoint run succeeds.

## Dependencies and Execution Order

- **Setup**: T001 must complete before installing the optional model extra.
- **User Story 1**: T002–T005 build the notebook path; T006 verifies the generated input and official feature output.
- **User Story 2**: T007 depends on the feature grid from T005; T008 depends on T006 and T007.
- **User Story 3**: T009–T010 can be prepared while model execution is in progress; T011 validates both execution paths after T002–T010.
- **Polish**: T012 depends on all previous tasks and successful evidence.

## Parallel Opportunities

- T002 and T003 can be developed in separate notebook sections once the notebook file exists.
- Documentation work in T010 can proceed independently from the correspondence code in T007, then must be reconciled in T012.
- Model download and CPU inference in T006/T008/T011 are sequential and may take substantial time.

## Implementation Strategy

Complete T001, then deliver User Story 1 as the minimum useful feature. Add the correspondence comparison in User Story 2. Complete resource documentation and evidence review in User Story 3. Run final alignment checks and mark the project roadmap task complete only when the measured checkpoint path succeeds.
