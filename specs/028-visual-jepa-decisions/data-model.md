# Data Model: Learned Visual JEPA Decision-Usefulness Lesson

All identifiers use fixed seeds and stable method names. Raw arrays are generated in memory. The result records use CSV and keep failures.

## Training transition

| Field | Type | Rule |
| --- | --- | --- |
| `training_seed` | integer | One of the five declared training seed groups. |
| `episode_id` | integer | Unique within the seed group. |
| `step` | integer | Zero-based simulator step. |
| `frame_t_minus_1`, `frame_t`, `frame_t_plus_1` | RGB byte images | Three consecutive resized frames. The model context is `[frame(t-1), frame(t)]`; its target context is `[frame(t), frame(t+1)]`. |
| `action` | float | Bounded to the MountainCar action range `[-1, 1]`. |
| `state_t`, `state_tp1` | two floats each | Position and velocity. Used only by the explicitly privileged state model and oracle. |

Evaluation episodes and intervention starts MUST use separate seeds and MUST NOT be included in this record.

## Intervention case

One held-out start paired with at least three candidate actions. The in-memory case has a two-frame observation context and simulator state. Saved CSV rows identify the start and action, record the actual position and method-specific prediction errors, and name the prediction space, endpoint-scoring space, device, model seed, and status. The RGB context is regenerated from the fixed start seed and identity; it is not serialized in the CSV. Action rankings are computed from each method's paired observed and predicted scores; they are not stored as a separate field. A visual latent or pixel score is a prediction-consistency measure, not task-reward accuracy. A failed prediction remains present with an error status and reason.

The saved intervention rows use the fixed method names `oracle`, `state_model`, `image_reconstruction`, and `jepa`. Oracle rows have an empty model seed and use the CPU simulator; other rows have an integer model seed and record the selected inference device. Prediction and scoring space labels are fixed by method. For each non-empty `(method, model_seed, start_id)` group, the actions `-1`, `0`, and `1` must each occur once. Every row must have finite `actual_position` and `observed_score`. Complete rows must also have finite prediction fields for their method; incomplete rows retain their status and error reason. A header-only file is valid in pilot mode, which does not collect intervention cases.

## Model run

One trained state or visual model, stored in `docs/evidence/visual-jepa-model-runs.csv`. It records method, training seed, training-data seed, objective name and settings, package versions, device, precision, training time, peak allocated GPU memory when available, final losses, embedding spread, collapse status, and run status. SIGReg fields remain specific to SIGReg. A VISReg-style row records its mix weight, scale/shape/center weights, projection count, epsilon, total regularizer, and component losses in separately named fields. The fixed `1e-3` low-variance threshold is defined in the feature spec and is not repeated in each CSV row. A collapsed or non-finite JEPA does not count as a successful model run.

## Evaluation episode

One method, optional model training seed, and one shared evaluation seed. It records status, success, return, steps, elapsed time, transition-model calls, planner settings, and failure reason. For learned methods, the model training seed stays distinct from the evaluation seed.

## Planner condition

The planner record identifies the model, candidate source (`uniform` or `scripted_momentum_prior`), action-sequence horizon, candidate count, elite count, update count, replanning rule, observation source, and scoring space. The two JEPA planner conditions share one trained checkpoint and differ only in candidate source.


## Validation-only pilot record

A pilot record captures one predeclared development data/model pair, run status, objective name and settings, split, device, versions, runtime, memory, validation prediction loss, regularizer and component losses where applicable, embedding spread, and collapse status. It MUST state that no intervention or evaluation data were used. It also stores a hash of the training notebook code cells and model/loss modules. Full mode accepts only the latest record that matches the current protocol hash and locked settings and has `run_status=complete` and `collapsed=false`. A missing, stale, or failed record keeps final training and evaluation closed.

Historical diagnostic rows without a protocol hash remain in the pilot ledger for provenance. They are ineligible to open the full-mode gate.
