# Data Model: JEPA Action Fidelity v2

## Protocol Lock

- `protocol_version`: fixed string `visual-jepa-action-fidelity-v2`.
- `status`: `locked`, `validation_complete`, `heldout_open`, `planning_open`, or `closed`.
- `protocol_hash`: SHA-256 of the frozen preregistration file.
- `code_hash`: SHA-256 of the experiment implementation files used for the run.
- `limits`: GPU minutes, peak GPU memory, storage, and transition-call ceiling.
- `seeds`: separate lists for training data, model initialization, test trajectories, planning starts, and smoke.

## Model Run

- Identity: objective, model seed, data seed, status, and stage.
- Training: rows, epoch count, optimizer settings, device, runtime, peak allocation.
- Loss: predictive, regularizer, total, and any objective-specific components.
- Gradients: encoder and predictor gradient norms per loss component and total; encoder gradient cosine between predictive and regularization terms.
- Representation: finite flag, feature mean and standard deviation, mean embedding norm, covariance trace, effective rank, and per-dimension spread summary.
- Probe: selected alpha, train-only feature scale, validation errors and R², Gram condition, coefficient norms, and bootstrap prediction variation.

## Action Intervention

- Identity: method, model seed, data seed, start ID, split, and status.
- Actions: exactly -1, 0, and +1 on the same visual context.
- Actual: simulator next position and velocity.
- Predicted: latent vector displacement, action derivative, raw decoded state, bounded state, and predictor norm.
- Diagnostics: position and velocity span error, pairwise order result, clipped flag, saturated flag, and non-finite flag.

## Rollout

- Identity: model, seed pair, trajectory seed, start ID, action-sequence hash, and status.
- Measurements: horizon, actual state, predicted state, latent error, position/velocity MAE, clipping, saturation, and out-of-distribution status.
- Missing horizon: explicit unsupported reason; no imputation.

## Planning Episode

- Identity: method, objective, model seed, evaluation seed, initialization, and status.
- Outcome: success, return, steps, transition calls, forward batches, planning time, full elapsed time, and device.
- Access: `visual`, `privileged_state`, or `exact_simulator`.
- Failure: reason stored as text; failed rows remain in output.

## Evidence Integrity

- Every CSV record includes protocol version and stage where useful.
- Every run summary includes row counts, file SHA-256 values, code hash, protocol hash, runtime, peak memory, and stop status.
- V1 raw evidence is read-only input and is never copied over or rewritten by the v2 runner.
