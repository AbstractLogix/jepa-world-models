# Data Model: LeHami Seed-0 Actuated Rollout Evaluation

No persistent application data is introduced. The entities below describe the run evidence.

## Run Record

- `model`: LeHami `a0q` or LeWM `b0s`.
- `training_seed`: seed 0.
- `checkpoint_sha256`: digest of the saved full-training checkpoint.
- `source_revision`: official LeHami git revision.
- `dataset_sha256`: digest checked against the official manifest.
- `manifest`: official episode-level split.
- `probe`: K=1-fitted 25-neighbor probe.
- `rollout_seed`: evaluator sampling seed 0.
- `horizon`: 76 rollout steps.
- `device` and `runtime`: actual execution environment.
- `wall_time_seconds`: measured evaluator wall time.
- `raw_output_directory`: unique path outside the project.

Validation: do not overwrite existing output directories; stop before evaluation if the source, checkpoint, or dataset hash does not match the recorded values.

## Per-Step Probe Metric

- `model`, `split`, `step`, `target`, `probe`, and `seed` identify the measurement.
- `mae_omega`: held-out angular-velocity mean absolute error at the named rollout step.
- `sample_count`: number of held-out episode anchors used for the row.

Validation: use test-split rows for paper comparison; retain the split label; reject missing or non-finite values; never pool the one-seed local values with the paper's three-seed statistics.
