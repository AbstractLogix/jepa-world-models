# Feature Specification: Validate Retained Episode Evidence

**Feature Branch**: `026-episode-evidence-validation`

**Created**: 2026-10-15

**Status**: Implemented

**Input**: Project goal to validate the shared evaluation protocol and retain inspectable local evidence.

## User Scenarios & Testing

### User Story 1 - Check retained planning records (Priority: P1)

A learner or reviewer wants to confirm that the saved Notebook 20 and Notebook 21 episode records still contain the complete paired evaluations described in the reports. One check should report success for valid records and identify the file and failed rule when a record is incomplete or malformed.

**Why this priority**: These two tracked CSV files are the only retained episode-level outputs for their lessons. The current audit checked them manually, so later edits could silently break their evidence.

**Independent Test**: Run the documented check on both committed CSV files. Confirm that each passes with 300 records, five groups, 100 paired starts, identical stored start states across methods, and all three methods per start. Run the validator against temporary copies with a deleted row, duplicate pairing key, mismatched start state, invalid status, and invalid numeric value; each copy must fail with a useful error.

**Acceptance Scenarios**:

1. **Given** both tracked CSV files are valid, **When** a reviewer runs the documented check, **Then** it reports each file's row and pairing coverage as valid.
2. **Given** a CSV is missing, duplicated, or malformed, **When** the reviewer runs the check, **Then** it exits with a failure and names the affected file and violated rule.
3. **Given** a CSV passes the integrity check, **When** a reader interprets that result, **Then** the documentation says this confirms record structure only and does not reproduce or validate a paper finding.

### Edge Cases

- A file is empty, has a missing or extra column, or contains an additional blank row.
- A paired start is missing a method or contains a duplicate method row.
- Methods share a pairing key but record different initial states for that key.
- A group maps to multiple group seeds or a seed group has an unexpected number of starts.
- A boolean, status, or numeric field contains an unsupported value, `NaN`, or infinity.
- A planner-only metric is blank on a planner row, or is populated on a direct-control row where it does not apply.

## Requirements

### Functional Requirements

- **FR-001**: The project MUST provide one documented check for `docs/evidence/policy-noise-episodes.csv` and `docs/evidence/sampling-based-episodes.csv`.
- **FR-002**: The check MUST verify the exact expected columns, 300 records per file, unique method-by-start keys, five groups, 100 paired starts, all three expected methods for each start, and identical initial states across methods for each paired start.
- **FR-003**: The check MUST validate outcome values and require numeric measurements to be finite and within any documented bounds. It MUST allow planner-only metric fields to be blank only for direct-control rows where the notebook defines those fields as not applicable.
- **FR-004**: Invalid records MUST produce a nonzero result and an error that identifies the file and failed rule. The check MUST NOT modify evidence files.
- **FR-005**: The protocol and audit MUST state that a passing check confirms retained-record integrity only; it does not rerun a notebook, validate scientific conclusions, or reproduce a paper.
- **FR-006**: The check MUST use the existing Python environment and add no package dependency.

### Key Entities

- **Episode record**: One method's outcome for one paired evaluation start, with its group or seed and per-episode measurements.
- **Evaluation group**: A declared group of paired starts used to report variation in the illustrative notebook.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Both committed episode files pass one documented check and report 300 records, five groups, 100 paired starts, and three methods per start.
- **SC-002**: The check fails on each specified temporary corruption case and identifies the affected file and rule.
- **SC-003**: The protocol audit records the executed check and retains the illustrative, non-reproduction boundary.

## Assumptions

- The two CSV schemas and expected method/group values remain defined by their notebooks and reproduction reports.
- Runtime values may vary between notebook runs; validation checks their format and valid range, not equality with a saved value.
- The check is limited to these two retained files. Other lesson evidence has different formats and remains covered by its existing report-level audit.
