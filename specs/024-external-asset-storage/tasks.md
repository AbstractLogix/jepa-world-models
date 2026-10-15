# Tasks: External Asset Storage

**Input**: Design documents in `/specs/024-external-asset-storage/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/storage-paths.md`, and `quickstart.md`

## Phase 1: Setup

**Purpose**: Preserve the existing uv-managed environment and dependencies.

- [x] T001 Confirm the project keeps uv as its package manager and adds no dependency or lockfile change in `pyproject.toml` and `uv.lock`.

## Phase 2: Foundational

**Purpose**: Provide one path resolver for notebook datasets and checkpoints.

- [x] T002 Add `JEPA_ASSETS_DIR` validation and local defaults in `src/jepa_tutorial/storage.py`.

## Phase 3: User Story 1 - Store lesson data and checkpoints outside the checkout (Priority: P1)

**Goal**: Route every in-scope dataset and checkpoint path through the shared resolver.

**Independent Test**: With the variable unset, paths resolve under `notebooks/`; with a valid temporary root, dataset and checkpoint paths resolve beneath it; an invalid root fails with its configured path and no fallback.

- [x] T003 [US1] Update `notebooks/02_lejepa_from_scratch.ipynb`, `notebooks/03_rectified_lpjepa.ipynb`, `notebooks/04_vjepa_plus_v1_jepa.ipynb`, `notebooks/05_value_guided_planning.ipynb`, `notebooks/06_gradient_based_planning.ipynb`, `notebooks/09_vjepa21_dense_features.ipynb`, `notebooks/11_vl_jepa.ipynb`, and `notebooks/13_object_centric_lejepa.ipynb` to use `src/jepa_tutorial/storage.py` for data and checkpoint paths.

## Phase 4: User Story 2 - Keep Python and model download caches on the selected drive (Priority: P2)

**Goal**: Show WSL and PowerShell setup for uv, lesson assets, Hugging Face, and PyTorch paths.

**Independent Test**: Follow the platform examples in the quickstart and print the resolved data and checkpoint folders before starting a download.

- [x] T004 [US2] Document external storage and future-lesson path rules in `specs/024-external-asset-storage/quickstart.md`, `README.md`, `CURRICULUM_GUIDE.md`, `docs/CURRICULUM_COVERAGE.md`, and `docs/ROADMAP.md`.

## Phase 5: Polish and cross-cutting checks

**Purpose**: Check the final implementation against the storage contract.

- [x] T005 Run focused path-resolution, notebook-format, lockfile, and diff checks for `src/jepa_tutorial/storage.py`, the eight affected notebooks, `pyproject.toml`, and `uv.lock`.

## Dependencies and execution order

- Setup (Phase 1) precedes the foundational resolver.
- User Story 1 depends on the resolver. User Story 2 is documentation work and can proceed in parallel after the storage interface is fixed.
- Final checks depend on the resolver, notebook updates, and documentation updates.

## Parallel opportunities

- Notebook updates and learner-facing documentation touch separate files and can proceed in parallel once the resolver names are fixed.
- WSL and PowerShell examples share the same environment-variable contract.

## Implementation strategy

Deliver User Story 1 first so notebooks can use an external asset root. Then complete User Story 2 and run T005. Do not move existing data, checkpoints, environments, or caches.
