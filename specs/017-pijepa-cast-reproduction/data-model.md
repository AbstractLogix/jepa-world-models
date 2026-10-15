# Data Model: PiJEPA Reproduction Records

This feature does not add a runtime schema. These records define the minimum fields to preserve in a future result report and raw outputs.

## PaperMetric

- `paper`: paper title and version.
- `table`: table identifier (`Table 2`).
- `encoder`: `V-JEPA-2 ViT-L`.
- `method`: `PiJEPA`, `Octo-WM`, `Octo`, or `default MPPI`.
- `metric`: metric name and unit; primary metric is `ATE XY RMSE` in meters.
- `published_value`: paper-reported value, kept separate from local measurements.

## AssetRecord

- `role`: code, data, encoder, policy, world model, or evaluator.
- `source`: first-party URL and immutable revision or artifact identifier.
- `content_hash`: hash of the acquired file or `unavailable` before acquisition.
- `license_or_terms`: exact license or `unknown`; unknown terms block transfer.
- `size_bytes`: known size or `unknown`.
- `accessed_at`: date checked.

## EvaluationCase

- `case_id`: stable identifier from the official validation set.
- `order`: position in the fixed evaluator input list.
- `input_hash`: hash of the exact case input when available.
- `method_results`: per-method output for the same case.

## TrainingRun

- `method`, `encoder`, `seed`, and `config_revision`.
- `checkpoint_source` and `checkpoint_hash`.
- `host`, `accelerator`, `software_versions`, `elapsed_time`, and `status`.
- `deviations` and `failure_detail`, including missing runs.

## EvaluationRun

- `method` and `encoder`.
- `checkpoint_ids` for every policy, world model, and encoder used by the method.
- `evaluator_revision`, `run_seed`, and `case_manifest_hash`.
- `host`, `accelerator`, `software_versions`, `elapsed_time`, and `status`.
- `raw_output_path` and `raw_output_hash` for per-case results.
- `deviations` and `failure_detail`, including cases or methods that did not complete.

## ResultRecord

- `paper_metric`: reference to the separate paper value.
- `local_metric`: value calculated from raw local results, or `not run`.
- `case_manifest_hash`, `training_run_ids`, and `evaluation_run_ids`.
- `raw_output_path` for per-case records.
- `claim_status`: `not run`, `partial`, `exploratory`, or `reproduced`.
- `limitations`: unverified case-list match, missing seeds, asset gaps, or other deviations.

## Relationships and validation

- One `PaperMetric` identifies one method and metric in the selected paper table.
- One `TrainingRun` can produce a checkpoint used by one or more `EvaluationRun` records.
- One `EvaluationRun` uses one method, its checkpoint set, and the shared case manifest; it can evaluate frozen checkpoints with no associated `TrainingRun`.
- Every compared method uses the same case manifest and evaluation conditions.
- A result cannot be labeled `reproduced` while planned seeds or cases are missing, required assets are unverified, or the paper's evaluation list is unconfirmed.
- Paper-reported values are immutable reference data; local values are computed only from preserved local outputs.
