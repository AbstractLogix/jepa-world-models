# Data Model: Model Calibration and Planning Evidence

The notebook uses in-memory NumPy arrays and writes compact local CSV evidence. This document defines the records needed to reproduce its summaries.

## ModelRun

- `model_seed`: one of five independent run identifiers.
- `train_episode_seeds`: the ten reset seeds used to collect training transitions.
- `bootstrap_seeds`: the five bootstrap samples fitted for the ensemble.
- `model_config`: polynomial degree, ridge penalty, and state/action fields.
- `software_versions`, `platform`, `elapsed_seconds`, and `status`.

## CalibrationCase

- `model_seed`: model run that generated the forecast.
- `split`: `calibration`, `in_distribution_test`, or `policy_shift_test`.
- `episode_seed`, `step`, `action`, and `previous_state`.
- `observed_next_state`, `ensemble_mean`, and `ensemble_std`.
- `interval_scale`, `interval_low`, `interval_high`, and `covered` for position and velocity.

## PlanRecord

- `model_seed`, `evaluation_seed`, and `method`.
- `start_state` and the selected bounded action sequence.
- `predicted_mean_return`, `ensemble_return_std`, and `planner_score`.
- `simulator_open_loop_return` from the same start and sequence.
- `exploitability_gap = predicted_mean_return - simulator_open_loop_return`.
- `planner_model_calls` and `planner_seconds`.

This is the initial selected 8-step plan gap for one evaluation start. It is not the paper's full policy-level XGap.

## EpisodeRecord

- `method`, `model_seed` (empty for the direct baseline), and `evaluation_seed`.
- `status`, `success`, `episode_return`, and `steps`.
- `model_calls`, `planner_seconds`, and `elapsed_seconds`.
- `failure_detail` when the episode does not complete.

## SummaryRecord

- `metric`, `method`, and the independent `model_seed` unit.
- Per-run mean across the 20 paired evaluation starts.
- Mean, median, 95% Student-t interval, and run count across five independent model runs.
- Paired per-start difference against the direct baseline, summarized first within model run.
- For calibration metrics, separate in-distribution and policy-shift coverage and mean interval width.

## Validation rules

- Training, calibration, in-distribution test, policy-shift test, and final evaluation seeds are disjoint.
- The direct baseline is evaluated once per final start; duplicate rows are not counted as extra training runs.
- Each selected plan's predicted and simulator returns use the same start state and exact action sequence.
- Calibration scales are fit only on calibration episodes. No test split changes model, interval, or planner settings.
- A missing or failed row stays visible and prevents a complete-comparison claim.
