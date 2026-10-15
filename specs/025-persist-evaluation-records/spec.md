# Feature Specification: Persist Raw Evaluation Records

**Feature Branch**: `025-persist-evaluation-records`

**Created**: 2026-10-15

**Status**: Implemented

**Input**: User goal: keep the shared evaluation protocol and local lesson evidence aligned.

## User Scenarios & Testing

### User Story 1 - Keep raw records from local planning lessons (Priority: P1)

A learner or reviewer wants to inspect the episode-level results behind Notebooks 20 and 21. Running each notebook should save every planned episode to a small, readable CSV in the repository's evidence folder, and the reports should link to those records.

**Why this priority**: The evaluation protocol asks reports to link raw per-run values when they are available. Both notebooks currently create 300 in-memory records and discard them after summarizing.

**Independent Test**: Execute each notebook in a fresh CPU kernel. Confirm each output CSV has 300 rows, the expected fields, all five evaluation groups, all 100 paired starts, and all three methods for every start. Confirm the report and protocol audit link to the saved files.

**Acceptance Scenarios**:

1. **Given** Notebook 20 completes its fixed evaluation, **When** it writes the evidence record, **Then** `docs/evidence/policy-noise-episodes.csv` contains all 300 method-by-start rows and their outcomes, seeds, actions-derived metrics, and pairing keys.
2. **Given** Notebook 21 completes its fixed evaluation, **When** it writes the evidence record, **Then** `docs/evidence/sampling-based-episodes.csv` contains all 300 planner-by-start rows, including statuses, failures, costs, and pairing keys.
3. **Given** a reader opens either reproduction report, **When** they inspect the local result, **Then** they can open the raw CSV and identify its columns and row count.
4. **Given** either notebook runs without a repository-root working directory, **When** it saves evidence, **Then** it locates the repository and writes to the same tracked evidence path.

## Requirements

### Functional Requirements

- **FR-001**: Notebook 20 MUST save every generated episode record to `docs/evidence/policy-noise-episodes.csv`.
- **FR-002**: Notebook 21 MUST save every generated episode record to `docs/evidence/sampling-based-episodes.csv`.
- **FR-003**: The CSV files MUST retain method, evaluation group or seed, paired-start identity, start state, outcome status, and the notebook's per-episode metrics.
- **FR-004**: The notebooks MUST verify that the saved record count matches the planned run count before reporting completion.
- **FR-005**: The reproduction reports and evaluation-protocol audit MUST link the raw files and describe their scope accurately.
- **FR-006**: The change MUST preserve planner behavior, fixed seeds, aggregate calculations, and the illustrative (non-reproduction) claim labels.
- **FR-007**: The evidence files MUST contain only the small, generated toy-task records; do not add external datasets, model weights, or user data.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A fresh CPU execution of Notebook 20 creates exactly 300 records across 100 paired starts and three methods.
- **SC-002**: A fresh CPU execution of Notebook 21 creates exactly 300 records across 100 paired starts and three methods.
- **SC-003**: Each CSV retains the planned row count and required provenance and outcome fields, including failed outcomes.
- **SC-004**: The two reproduction reports and the protocol audit link to the corresponding saved CSV files.
- **SC-005**: No model, dataset, or training behavior changes, and the outputs remain explicitly illustrative.

## Assumptions

- The notebooks run from within a project checkout containing `pyproject.toml`.
- `docs/evidence/` exists in a clean checkout; evidence writes do not create a new dependency.
- Runtime measurements can differ between executions and remain raw observations, not deterministic assertions.
