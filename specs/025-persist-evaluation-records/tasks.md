# Tasks: Persist Raw Evaluation Records

**Input**: Design documents in `/specs/025-persist-evaluation-records/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/episode-records.md`, and `quickstart.md`

## Phase 1: Setup

**Purpose**: Keep evidence output dependency-free.

- [x] T001 Confirm that `docs/evidence/` is tracked and CSV generation uses the existing Python environment in `pyproject.toml` and `uv.lock`.

## Phase 2: Foundational

**Purpose**: No shared infrastructure is needed; each notebook owns its small CSV writer.

## Phase 3: User Story 1 - Keep raw records from local planning lessons (Priority: P1)

**Goal**: Retain every episode record from Notebooks 20 and 21 and link the CSVs from their reports.

**Independent Test**: Fresh CPU execution of each notebook creates exactly 300 rows with the documented schema, unique paired-start keys, all methods and groups, and no dropped failures. The reports and audit link to the files.

- [x] T002 [US1] Write all 300 Notebook 20 records to `docs/evidence/policy-noise-episodes.csv` with the schema in `specs/025-persist-evaluation-records/data-model.md` from `notebooks/20_policy_noise_planning.ipynb`.
- [x] T003 [P] [US1] Write all 300 Notebook 21 records to `docs/evidence/sampling-based-episodes.csv` with the schema in `specs/025-persist-evaluation-records/data-model.md` from `notebooks/21_sampling_based_planning.ipynb`.
- [x] T004 [P] [US1] Link the raw CSV files and describe their columns and scope in `docs/reproductions/policy-noise-planning.md`, `docs/reproductions/sampling-based-planning.md`, and `docs/CURRICULUM_COVERAGE.md`.
- [x] T005 [US1] Update `docs/evidence/evaluation-protocol-audit.md` to record the saved files and their verified row coverage.
- [x] T006 [US1] Execute both notebooks in fresh CPU kernels and validate schemas, 300 rows, unique pairing keys, five groups, three methods, and report links for `docs/evidence/policy-noise-episodes.csv` and `docs/evidence/sampling-based-episodes.csv`.

## Dependencies and execution order

- T001 is complete and confirms no setup change is needed.
- T002 and T003 can proceed in parallel because they edit different notebooks and produce separate files.
- T004 can proceed in parallel with the notebook edits because it changes only reports.
- T005 and T006 depend on T002–T004.

## Parallel opportunities

- Notebook 20 output work, Notebook 21 output work, and the report links touch separate files.

## Implementation strategy

Add persistence after each existing evaluation loop without changing planner logic. Update both reports and the audit, execute both notebooks, then inspect and validate the saved raw records.
