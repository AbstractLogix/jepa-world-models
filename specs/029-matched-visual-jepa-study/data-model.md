# Data Model: Matched Visual JEPA Decision Study v1

All records include `experiment_version=visual-jepa-decision-study-v1`. The versioned evidence files do not replace Notebook 22 records.

## Planner development row

One exact-dynamics CEM configuration on one development start. Store planner settings, stage, setting ID, seed, initial state, status, success, return, steps, rollout-transition calls, elapsed and planning time, device, and failure reason. Keep all attempted settings, including failures.

## Protocol lock

JSON with schema version, experiment version, selection timestamp, selected horizon/candidate/elite/update/score/initialization/replan values, development success counts, confirmation result, search-grid identity, code hash, preregistration hash, call budget, and status. `status=locked` is valid only when the confirmation criteria pass.

## Oracle gate row

One exact-dynamics episode from one final evaluation start using the locked protocol. Store protocol-lock hash, start seed/state, status, success, return, steps, calls, wall time, and error. The gate record stores the count and threshold. No learned-model test row may be written if the oracle gate fails.

## Training run

One model family and one model/data seed pair. Store episode split counts, training/validation rows, objective, model state hash, probe hyperparameter, probe validation errors, device/software versions, training time, peak GPU memory, model-native losses, collapse metrics, and status. Learned state, reconstruction, and JEPA methods share each data seed group.

## Probe test row

One model seed and independent test trajectory group. Store test-row identity, position MAE/R2, velocity MAE/R2, action-effect fidelity, finite status, and probe hash. Test labels are never used to select or fit the probe.

## Intervention row

One method, model seed (empty only for oracle), held-out state, and action from `{-1, 0, 1}`. Store actual and predicted next position/velocity, errors, predicted and observed common continuation return, scoring-space label, method access, device, status, and failure reason. Actions for a state remain paired. Action ranking uses common task returns; native-space metrics remain separate.

## Episode row

One method/condition, optional model seed, and shared evaluation-start seed. Store initialization (`uniform`, `momentum_prior`, or `direct`), privileged access, status, success, Gymnasium return, steps-to-goal or episode limit, candidate rollout transitions, actual model calls, planning time, total elapsed time, device, and error. Direct control has one row per start and is not duplicated across model seeds.

## Figure manifest

For every generated figure, store source file hashes, filter/method labels, sample sizes, aggregation and interval method, evidence status, output path, and code hash.
