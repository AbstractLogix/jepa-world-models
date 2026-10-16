# Research Notes: Supervised Visual Positive-Control Study v3

**Frozen input**: PR #6 head `bbafed5`, reviewed against PR #5 head `fe57c2e0`
**Date**: 2026-10-16
**Scope**: Local code, v2 protocol, and retained v2 raw evidence. No v3 model was trained for this audit.

## Verified v2 implementation

- `src/jepa_tutorial/action_conditioned_visual.py` defines a six-channel visual encoder with convolutional blocks, adaptive pooling, a 32-dimensional projection, and `LayerNorm(elementwise_affine=False)`.
- `src/jepa_tutorial/visual_mountaincar.py` saves previous, current, and next resized 48 × 128 RGB frames. The v2 runner supplies previous and current frames to the model and supplies the action separately.
- Inputs are converted from uint8 to float and divided by 255. No image mean/std normalization is applied.
- `SupervisedVisualDynamics` appends the scalar action to the 32-dimensional encoder output, then uses a 33 → 64 → 2 MLP. It predicts absolute next position and velocity.
- The supervised loss is raw equal-weight MSE over the two output coordinates. Training uses AdamW (0.002 learning rate, 0.0001 weight decay), batch size 256, 35 epochs, and the first 16 episodes for training with the last four for validation.
- The v2 action test holds the same two-frame context fixed while it predicts actions -1, 0, and +1. Raw predictions are retained before task-bound clipping.

## Verified v2 measurements

The following values come from retained run and action records, not a summary-only report:

| Data/model seed | Probe position R² | Probe velocity R² | Embedding effective rank | Direct next-position MAE | Direct next-velocity MAE | Position / velocity contrast error | Position / velocity order accuracy | Clip fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9301 / 9401 | 0.914 | -0.0004 | 1.011 | 0.0343 | 0.00704 | 1.301 / 0.688 | 1.000 / 0.950 | 0.000 |
| 9302 / 9402 | 0.930 | -0.0004 | 1.020 | 0.0261 | 0.00424 | 1.415 / 0.573 | 0.467 / 0.500 | 0.000 |
| 9303 / 9403 | 0.968 | 0.0018 | 1.161 | 0.0803 | 0.00545 | 0.236 / 0.733 | 1.000 / 0.667 | 0.000 |

`position_mae` and `velocity_mae` in the run table are linear-probe errors. `one_step_position_mae` and `one_step_velocity_mae` are direct action-conditioned transition errors. These metrics are not interchangeable.

The v2 observation audit reports handcrafted state errors of 0.003747–0.003751 position and 0.0000555–0.0000574 velocity across its three groups. Stored transition labels match the simulator to less than 6e-8. Handcrafted state plus exact dynamics has action-order accuracy 1.0 on all 20 validation contexts per group.

On the v2 training partitions regenerated with the retained collector, raw next-state variance was:

| Data seed | Position variance | Velocity variance | Position / velocity variance ratio |
|---:|---:|---:|---:|
| 9301 | 0.0082107 | 0.00005826 | 140.9 |
| 9302 | 0.0080146 | 0.00005740 | 139.6 |
| 9303 | 0.0083612 | 0.00005960 | 140.3 |

The v2 aggregate gradient snapshots are nonzero but fluctuate. Predictor/encoder gradient-norm median ratios are 6.56, 2.88, and 7.85 for the three supervised seeds. The modules have different parameter counts, and the records are aggregate norms. These values do not show a defective optimizer or identify per-coordinate suppression.

## Decisions and alternatives

| Decision | Rationale | Alternative not selected |
|---|---|---|
| Keep the v2 model architecture and raw loss as the main visual dynamics baseline. | This preserves a direct comparison to the failed supervised control. | Rebuilding its architecture would prevent isolating target effects. |
| Add one train-standardized-target arm. | The v2 target-variance ratio is large and raw MSE does not balance coordinates. This tests one factor only. | Do not change the encoder, optimizer, image input, and loss weights in the same arm. |
| Add one residual-output arm with a fixed handcrafted current-state skip. | It tests a residual target with the learned encoder and action head held fixed. | A jointly trained current-state and delta model would change architecture and objective at once. |
| Keep an action-shuffled copy of the raw-target baseline. | It tests whether the model uses correctly paired action labels. | Do not shuffle validation actions, frames, or targets. |
| Keep the fixed ridge regressor deliberately simple. | It asks whether a linear map from `[x, v, action]` is sufficient. A failure does not imply that the features lack information. | Do not add handpicked nonlinear physics features or tune alpha on validation. |
| Require a minimal physical representation check before opening independent tests. | The v3 task has two observed state coordinates. Require at least two varying embedding dimensions, effective rank ≥2.0, and both linear state probes R² ≥0.80. This is weaker than the v2 JEPA rank cutoff and matches this small state target. | Do not require rank 8 for a two-coordinate supervised positive control; that could reject a compact representation that preserves both state variables. |
| Do not add another neural architecture or a JEPA objective. | The requested study is a positive-control diagnosis, not a broader tutorial or planner comparison. | New architecture search and JEPA tuning are out of scope. |

No paper result or new literature claim is needed for this experiment. It is a local controlled study of a synthetic simulator.
