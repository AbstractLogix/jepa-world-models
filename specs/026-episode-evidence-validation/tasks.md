# Tasks: Validate Retained Episode Evidence

**Input**: Design documents in `/specs/026-episode-evidence-validation/`

**Prerequisites**: `plan.md` and `spec.md`

## Phase 1: User Story 1 - Check retained planning records (Priority: P1)

**Goal**: Make the saved Notebook 20 and Notebook 21 records easy to validate without changing their contents.

**Independent Test**: The documented command passes on both tracked CSVs with 300 rows each, five groups, 100 paired starts, matching initial states, and three methods per start. Temporary copies with a removed row, duplicate key, mismatched start state, invalid status, or malformed number fail with a file-specific message. The audit labels the result as record-integrity evidence only.

- [x] T001 [US1] Implement the standard-library validator in `scripts/validate_episode_evidence.py` for the schemas and invariants listed in `specs/026-episode-evidence-validation/plan.md`.
- [x] T002 [US1] Run the validator against both retained files and verify its failure paths with temporary corrupted copies of `docs/evidence/policy-noise-episodes.csv` and `docs/evidence/sampling-based-episodes.csv`, including changed paired start values in both records.
- [x] T003 [US1] Document the command and its evidence boundary in `docs/EVALUATION_PROTOCOL.md` and record the executed result in `docs/evidence/evaluation-protocol-audit.md`.
- [x] T004 [US1] Align `specs/001-project-vision/alignment.md`, `specs/001-project-vision/remaining-work.md`, and `specs/001-project-vision/tasks.md` with the new validator and its limited scope.
- [x] T005 [US1] Analyze `specs/026-episode-evidence-validation/spec.md`, `plan.md`, and `tasks.md`; then converge the feature against the validator, protocol, audit, and project records, appending only verified in-scope gaps to `specs/026-episode-evidence-validation/tasks.md`. No in-scope gaps remained.

## Dependencies and Execution Order

- T001 precedes T002 and T003.
- T002 and T003 precede T004.
- T005 is the final feature check after implementation and alignment.

## Implementation Strategy

Implement the two focused validators first. Confirm they accept the saved rows and reject malformed copies. Then update the protocol and audit, align project status records, and run Spec Kit analysis and convergence.
