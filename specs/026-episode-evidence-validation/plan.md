# Implementation Plan: Validate Retained Episode Evidence

**Branch**: `026-episode-evidence-validation` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature to check the retained Notebook 20 and Notebook 21 episode records with a repeatable local command.

## Summary

Add a small, dependency-free validator for the two tracked episode CSV files. Check their schemas, counts, pairing keys, groups, methods, and per-row values. Document the command and its claim boundary in the shared protocol and evidence audit. Do not change notebook behavior or paper-evaluation scope.

## Technical Context

**Language/Version**: Python 3.10–3.13.

**Primary Dependencies**: Python standard library only.

**Storage**: Read the two tracked CSV files in `docs/evidence/`; do not write to them.

**Testing**: Run the command on the committed files and call its validators with temporary corrupted copies to check failure behavior.

**Target Platform**: Linux, macOS, and Windows project checkouts.

**Project Type**: Educational Python notebooks and evidence documentation.

**Constraints**: No notebook execution, external downloads, model runs, new dependencies, or changes to evaluation results. A passing check is record-integrity evidence only.

## Constitution Check

- **Teach from evidence**: Pass. The report labels remain illustrative.
- **Measure before making claims**: Pass. The check validates retained raw rows and does not calculate new performance claims.
- **Keep changes small and reviewable**: Pass. One standard-library script reads two fixed files.
- **Add tools only when a lesson needs them**: Pass. The evaluation audit already re-reads and checks these records manually; the script makes that check repeatable.
- **Keep uncertainty explicit**: Pass. The command reports data integrity, not scientific validity or paper reproduction.

## Project Structure

```text
scripts/validate_episode_evidence.py
docs/evidence/policy-noise-episodes.csv
docs/evidence/sampling-based-episodes.csv
docs/EVALUATION_PROTOCOL.md
docs/evidence/evaluation-protocol-audit.md
specs/001-project-vision/alignment.md
specs/001-project-vision/remaining-work.md
specs/001-project-vision/tasks.md
```

**Structure Decision**: Keep the validator in `scripts/`, use the repository root derived from the script location, and expose validation functions that accept explicit paths so temporary copies can test rejection behavior. The command-line entry point uses the two tracked paths by default.

## Validation Rules

- Notebook 20 exact columns: `method`, `evaluation_seed`, `start_id`, `start`, `success`, `steps`, `model_calls`, `final_position`, `mean_selected_action_error`, and `off_support_fraction`. Require 300 rows; methods `raw_action`, `policy_noise`, and `direct_control`; seeds 101, 202, 303, 404, and 505; 20 start IDs (0–19) per seed; one row for every method at each seed/start pair; the numeric `start` value must match across methods for the same pair; `success` is `True` or `False`; integer step and call counts are nonnegative; other measurements are finite; prediction error is nonnegative; and `off_support_fraction` is in [0, 1]. The two planner-only metrics must be present for planner rows and blank for direct-control rows, as defined in `specs/025-persist-evaluation-records/data-model.md`.
- Notebook 21 exact columns: `method`, `group`, `group_seed`, `paired_start`, `start_x`, `start_y`, `success`, `status`, `steps`, `transition_calls`, `runtime_seconds`, `final_x`, and `final_y`. Require 300 rows; methods `RRT`, `KPIECE`, and `CEM`; groups 1–5 with fixed seeds 2101–2105 and 20 paired starts each; one row for every method at each group/start pair; `(start_x, start_y)` must match across methods for the same pair; statuses are `success`, `episode_limit`, or `invalid_planned_action`; `success` matches status; integer step and transition-call counts are nonnegative; and other measurements are finite and nonnegative where they represent runtime or counts.
- Both checks accept ordinary episode failures as valid outcomes. They reject missing rows, duplicate keys, unknown values, malformed numbers, and invalid row counts.

The validator functions accept a CSV path. The command-line entry point checks the repository's two default evidence paths, prints a concise result for each, and returns a nonzero status if either file fails. Errors include the file and the failed field or invariant.

## Documentation

Add the command near the evaluation protocol's raw-record guidance. Update the audit with current command output and state explicitly that the result does not certify metrics, notebook execution, or a paper finding.
