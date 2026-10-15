# Data Model: LeHami Actuated-Pendulum Velocity Reproduction

## Dataset and split

| Field | Value |
| --- | --- |
| Dataset | Generated actuated pendulum, 900 episodes |
| Frames | 64×64 RGB, 80 frames per episode |
| Design cells | Rod lengths 0.7, 1.0, 1.4 m; masses 0.5, 1.0, 2.0 kg |
| Actuation | Torque held constant for 3 frames |
| Split | 630 train / 135 validation / 135 test, at episode level |
| Generator/split seed | 42 |
| Integrity | Generated HDF5 SHA-256 must match `dataset_sha256` in the shipped manifest |

## Model run

Each run records model arm (`a0q` or `b0s`), training seed (0, 1, or 2), official config path, code revision, checkpoint path and hash, training status, elapsed time, device, and peak GPU memory when available.

## Probe measurement

The target is angular velocity in rad/s. The probe is the official distance-weighted 25-neighbor regressor, fitted to K=0 training features and scored on K=0 features from held-out test episodes. Preserve the per-run MAE and the raw official probe output. The aggregation is the arithmetic mean and sample standard deviation across the three training seeds, computed separately for each model.

## Reproduction report

The report records paper-reported values separately from local per-seed values and aggregates. It also records code and dependency revisions, manifest and dataset hashes, hardware/software versions, wall time, missing runs, deviations, and the final status (`reproduced selected result`, `partial`, or `not run`).
