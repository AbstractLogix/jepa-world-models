# Supervised Visual Positive-Control Study v3

**Status**: Draft preregistration; no v3 model training has started
**Date**: 2026-10-16
**Starting point**: Draft PR #6, `bbafed5`
**Spec Kit feature**: [Feature 031](../../../specs/031-supervised-visual-positive-control-v3/spec.md)
**Prior evidence**: [v2 investigation](../visual-jepa-action-fidelity-v2-investigation.md)

## Pre-training amendment

On 2026-10-16, static review found two evidence-completeness gaps before training. The validation runner now records every planned arm and a stop reason when the data-path gate fails. The held-out protocol and runner record each scheduled start, including missing starts. No training or independent test data had been accessed. This amendment does not change the methods, data, thresholds, or selection rules.

## Question and scope

Can a small supervised visual model recover current position and velocity, then predict action-conditioned next states with useful per-coordinate accuracy?

This study investigates the supervised visual-control failure in v2. It does not evaluate JEPA objectives, train a policy, run planning, or change Notebook 22, Notebook 23, or any v1/v2 record. The target claim is local to this synthetic MountainCar task.

## Pre-study audit and competing explanations

The v2 model receives two consecutive 48 × 128 RGB frames plus the action. Frames are ordered previous then current, converted to float in [0, 1], and concatenated into six channels. It predicts absolute next position and velocity with a 32-dimensional VisualEncoder and a 33 → 64 → 2 action head. It uses raw equal-weight MSE, AdamW at 0.002 with 0.0001 weight decay, batch size 256, and 35 epochs.

The saved v2 run records show that all three supervised visual fits completed but failed validation. One-step position MAE is 0.0261–0.0803; velocity MAE is 0.00424–0.00704. Position action-order accuracy is 0.467–1.0 and velocity action-order accuracy is 0.50–0.95. Normalized contrast error is 1.301–1.415 for position in two seeds and 0.236 in one; it is 0.573–0.733 for velocity. No supervised visual action prediction clipped. The model is not uniformly action-insensitive: several runs preserve action order, but their effect magnitudes and state errors fail the frozen gates.

The observation-control evidence confirms that the two-frame input contains the needed signal. Across v2 data groups, the fixed color-center estimator has position MAE 0.003747–0.003751 and velocity MAE 0.0000555–0.0000574. The simulator transition matches stored labels within 6e-8. On the 20 validation contexts per group, handcrafted state plus exact dynamics passes action-order and contrast checks.

The v2 learned embeddings have effective rank 1.01–1.16. Linear position-probe R² is 0.914–0.968; velocity-probe R² is approximately zero. Encoder norms near √32 are imposed by the final per-sample LayerNorm and do not show between-example spread. V2 gradient snapshots report aggregate encoder and predictor norms, not coordinate-specific gradients; they cannot establish whether velocity gradients are suppressed.

Regenerated v2 training labels show next-position variance of 0.00799–0.00836 and next-velocity variance of 0.0000570–0.0000596. The ratio is 139.6–140.9. The validation ratio is 126.6–138.3. One-step action effects are small in absolute units. This supports target-scale imbalance as a plausible cause. It does not establish causation: v2 did not compare raw and train-standardized targets under an otherwise matched protocol.

| Explanation | Current evidence | Status before v3 |
|---|---|---|
| Missing temporal information | Two frames are present; their handcrafted difference recovers velocity accurately. | Not supported as a data-path cause; a learned encoder may still fail to use it. |
| Incorrect renderer, split, or labels | Handcrafted features and exact simulator transitions pass retained controls. | Not supported for v2. Recheck on the new v3 data. |
| Weak learned velocity representation | Probe velocity R² is near zero; covariance rank is near one. | Confirmed as a model diagnostic, not a unique root cause. |
| Absolute target scale imbalance | Position target variance is about 140 times velocity target variance under raw MSE. | Plausible; test with one standardized-target ablation. |
| Action input ignored | Some seeds preserve action order; others are near chance. Contrast magnitudes fail in every seed. | Mixed. Measure input sensitivity and compare with an action-shuffled control. |
| Clipping or saturation | No supervised visual predictions clipped in v2. | Not supported as a v2 cause. Measure again. |
| Optimizer or gradient failure | Aggregate gradients are nonzero and variable; coordinate gradients are absent. | Unresolved. Record per-coordinate gradients without changing optimizer settings. |
| Linear/nonlinear mapping limitation | The visual encoder and small action head may underfit the transition. | Hypothesis. Compare a fixed linear feature regressor, learned current-state estimator, and matched CNN controls. |

## Root-cause investigation order

1. Recreate only the three preregistered v3 training/validation data groups. Check episode isolation, target scales, frame preprocessing, two-frame state signal, and exact transition labels. Do not create test-seed data.
2. Run the exact simulator control and fixed handcrafted state estimator on training and validation examples. Stop if either frozen data-path gate fails.
3. Fit a fixed ridge regressor on the same handcrafted `[position, velocity, action]` features. This tests a simple map; failure alone does not show the visual information is insufficient.
4. Fit a learned visual current-state estimator. Record position and velocity errors and embedding diagnostics. This separates state extraction from next-state prediction.
5. Fit the existing absolute-state model unchanged, then the one-factor target-standardization ablation, residual output model, and action-shuffled negative control. Use matched seed pairs and data throughout.
6. Compare per-coordinate state loss, action-contrast error, order, sensitivity, clipping, gradients, and learning curves. Attribute a target-scale effect only if the matched standardized-target arm improves the frozen outcomes without changing other settings. Attribute action learning only if the unshuffled model passes and the shuffled control fails.
7. Apply the frozen validation gate. If it fails, retain the negative result and stop. If it passes, the selected supervised model may be evaluated on the reserved test seeds and multi-step trajectories. Do not run a planner in this feature.

## Task and data access

- Environment: the repository's synthetic `MountainCarContinuous-v0` variant with position bounds [-1.2, 0.6], velocity bounds [-0.07, 0.07], actions in [-1, 1], and a 300-step episode limit.
- Training and validation groups: data seeds 9701, 9702, 9703; 20 random-action episodes per seed. Episodes 0–15 are training, episodes 16–19 are validation.
- Model seeds: 9801, 9802, 9803, paired by index with the data seeds. All neural methods use the same model/data pair.
- Validation contexts: 20 distinct rows sampled without replacement from episodes 16–19 per data seed using the SHA-256 stable-seed helper keyed by this protocol version, role `validation-action-contexts`, and data seed. Share each context across all methods.
- Reserved independent test groups: seeds 9911–9915, with contexts at steps 40, 90, 140, and 190 from the seeded evaluation trajectory. Do not create or inspect these groups unless the validation gate opens.
- No external data, pretrained model, or robot simulator is used. Training and validation trajectories are generated locally and transiently.

## Fixed model hierarchy

All visual methods use the same stored two-frame pair, 48 × 128 RGB preprocessing, [0, 1] input scale, episode split, paired data seed, and validation contexts. The two frames remain ordered previous then current.

1. **Handcrafted state plus exact dynamics**: extract current position from the red-car color center and velocity from the difference between current and previous extracted positions. Apply the environment's exact transition to this estimated state and each action. This is a diagnostic reference, not a learned model.
2. **Handcrafted-feature supervised regression**: fit multi-output ridge regression with fixed alpha 1.0 from train-standardized `[estimated position, estimated velocity, action]` features to absolute next position and velocity. Fit feature means and scales on training rows only. Do not add interactions or tune alpha.
3. **Learned visual current-state estimator**: use the existing 32-dimensional `VisualEncoder` on the two frames and a linear 32 → 2 head to predict the current simulator position and velocity. Use raw coordinate targets and equal-weight MSE. This model has no action input and is a state-estimation control, not a dynamics model.
4. **Existing supervised visual dynamics**: preserve the v2 `VisualEncoder(32)` and `Linear(33, 64) → GELU → Linear(64, 2)` head. Concatenate the scalar action to the encoded two-frame context. Predict absolute next position and velocity with raw equal-weight MSE.
4a. **Target-scale ablation**: use the exact model and training settings in method 4. Standardize the two next-state targets by the training-split mean and standard deviation (minimum scale 1e-6), use equal-weight MSE in standardized target space, and invert the transform at inference. This is the only change from method 4.
5. **Residual dynamics**: use the exact encoder and action head in method 4. Predict the residual from the fixed handcrafted current-state estimate to the absolute next state: `next_state - handcrafted_current_state`. Add the prediction to that same handcrafted current-state estimate at inference. Do not normalize residual targets. This is a hybrid visual-model diagnostic, not a pure learned visual state model.
6. **Action-shuffling negative control**: use method 4 unchanged, but permute action labels within the training split using a deterministic shuffle seed. Do not shuffle frames or targets. Keep validation actions and evaluation action triplets correct.

Methods 3, 4, 4a, 5, and 6 are the 15 neural fits. Method 2 is fit once per paired data group. Methods 1 and the exact privileged-state calculation have no learned parameters. The final epoch is used for validation; do not select a checkpoint by validation loss.

## Training and diagnostics

- Neural methods: float32, AdamW, learning rate 0.002, weight decay 0.0001, batch size 256, 35 epochs, no early stopping. Train batches are shuffled with the fixed model seed. Fix PyTorch, NumPy, and device seeds before initialization.
- Method 4 retains the v2 architecture and raw-target loss. Method 4a changes only train-only target standardization. Method 5 changes only the target to a residual and adds the fixed handcrafted state skip. Method 6 changes only training action pairing.
- Save per-epoch train and validation objective loss, raw position MSE, raw velocity MSE, and decoded state errors. Record gradient norms for the position and velocity components separately on a fixed training batch at epochs 1, 5, 10, 15, 20, 25, 30, and 35.
- Record target mean, standard deviation, range, and variance ratio by coordinate; encoder spread, effective rank, and output norm; linear state probes; probe conditioning; and raw-to-clipped state counts.
- Evaluate every action-conditioned method at actions -1, 0, +1 for each of the same 20 validation contexts. Retain raw predicted states before clipping and actual environment next states.
- Record one-step MAE per coordinate, action contrast per coordinate, pairwise ordering, predicted action sensitivity, action input derivative at action zero, finite status, and clipping rate. The current-state estimator records state-estimation errors and has no action-fidelity score.
- Report each seed pair separately. With three pairs, provide median and range only; do not treat the 60 action rows as independent seed replicates.

## Frozen validation thresholds and selection

### Data-path gate

For every data seed, the exact simulator must match stored transition labels to maximum absolute error ≤1e-6. The handcrafted visual state plus exact dynamics must have full-validation position MAE ≤0.005 and velocity MAE ≤0.0002. On the 20 shared action-triplet contexts it must also have position and velocity contrast error ≤0.01, pairwise order accuracy ≥0.95, at least 18 valid starts per coordinate, and clipping fraction ≤0.05. If any condition fails for any data seed, do not fit learned models; report a data-path failure.

### Per-seed learned dynamics gate

A method 4, 4a, or 5 seed passes only if all are true on its shared validation data:

1. All training and validation losses and predictions are finite.
2. The validation embedding has no more than 30 of 32 dimensions with standard deviation below 1e-3, effective rank ≥2.0, and train-fitted linear state probes with validation R² ≥0.80 for both position and velocity. The probe standardizes features using training rows only, tests the fixed ridge penalties 0.01, 0.1, 1.0, and 10.0, and selects by validation MSE. These thresholds require at least two varying representation dimensions and independently readable state coordinates; they do not reuse the stricter JEPA rank threshold from v2.
3. Raw next-state position MAE ≤0.01 and velocity MAE ≤0.002.
4. Median per-context normalized action-contrast error ≤0.25 for both coordinates. For each context, average the absolute errors in predicted action-minus-zero contrasts for actions -1 and +1, then divide by the actual three-action span with a 1e-4 floor.
5. Pairwise action-order accuracy ≥0.80 for both coordinates. Exclude actual pairwise differences of magnitude ≤1e-5. Each coordinate has at least 12 valid contexts.
6. No more than 5% of the 60 raw next-state predictions are outside the simulator state bounds. Score the raw state; retain any clipped copy only as a separate environment-consistent diagnostic.

Record action sensitivity even when a method fails. A zero or incorrectly directed response fails the contrast/order gate; do not use a post-hoc sensitivity threshold.

### Family gate and selection

- A learned dynamics family is eligible only if at least two of its three independent model/data pairs pass every per-seed learned dynamics gate.
- The action-shuffling control is valid only if it fails the action-fidelity checks on at least two pairs. If it passes the learned action-fidelity gate on two or more pairs, mark the study inconclusive and keep the test stage closed.
- If one or more of methods 4, 4a, and 5 are eligible, select by lowest median across passing seeds of the mean position/velocity normalized contrast error; then lowest mean of position MAE / 0.01 and velocity MAE / 0.002; then fixed order 4, 4a, 5. Do not use test records for selection.
- Open test data only if all data-path checks pass, one candidate family is eligible, the action-shuffling control is valid, all required validation records are complete, and the protocol/evidence lock verifies.
- If no family is eligible or any gate is invalid, record `heldout_open=false`. Do not create test-seed data, multi-step rollouts, or planning outcomes.

## Conditional held-out stage; no planning

If the frozen validation gate passes, evaluate the selected family on all five reserved test groups. Use all three trained model seeds and the shared 20 test contexts.

1. Include exact dynamics, handcrafted state plus exact dynamics, the fixed ridge baseline, and the learned current-state estimator as controls.
2. For each context, draw 60 actions from U[-1, 1]. Use the SHA-256 stable-seed helper with the protocol version, test seed, context ID, and role `heldout-rollout-actions`. Give every dynamics method the same action sequence.
3. Create each test start trajectory with `reset(seed=test_seed)`. Use a U[-1, 1] action stream from a stable seed keyed by the protocol version, role `heldout-start-trajectory`, and test seed. Retain starts at steps 40, 90, 140, and 190.
4. If the simulator ends before a scheduled start, record the missing start and mark held-out coverage incomplete. Do not replace the seed or start.
5. At each start, use the same four 60-step action sequences for every method in that test seed. Start from the recorded test state and two frames. Render each predicted position for the next visual input.
6. Compare exact dynamics, handcrafted state plus exact dynamics, fixed ridge, and the selected visual dynamics model at horizons 1, 5, 10, 20, and 60.
7. Report raw position and velocity error, error growth, clipping, rollout completion, and the fraction of predicted states outside the training partition's minimum-to-maximum position and velocity range. Report every seed and context, including failed predictions.

These are held-out predictive dynamics results. They are not planning or policy results. Do not run CEM, train a policy, or open JEPA planning in v3.

## Failure, abstention, and compute limits

- Maximum training: 15 neural fits, 35 epochs each; 60 GPU-minutes total.
- Maximum allocated CUDA memory: 2 GiB. Stop the training stage if exceeded.
- Maximum transient output on E:: 5 GiB, including checkpoints and generated arrays. Do not commit datasets, frame tensors, or checkpoints.
- Stop without seed replacement if a required fit is incomplete, non-finite, or exceeds the compute limit. Preserve its status and error.
- If the data-path gate fails, retain the exact-dynamics outcome and one `not_run` record with a stop reason for every later planned arm.
- Do not change the architecture, loss, target transform, data, thresholds, or selected seed after viewing validation results.
- A failed data-path or action-shuffle sanity gate blocks independent test data.
- Planning is out of scope even if held-out prediction passes.
- Before the first neural fit, freeze this protocol and a lock containing its SHA-256 and the study code hashes. Every run validates the lock.

## Evidence and figures

Retain compact CSV/JSON evidence under `docs/evidence/visual-jepa-supervised-positive-control-v3/`: protocol lock; partition map; target scales; data-path checks; exact and ridge baselines; row-level validation predictions; per-dimension embedding mean, variance, and spread; training curves; per-coordinate gradient records; state probes; action triplets and sensitivities; per-seed gates; run summary; and a SHA-256/byte-size manifest. If the gate opens, also retain every held-out start and rollout row, including failures. Save transient data and checkpoints on E: under `visual-jepa-supervised-positive-control-v3/`.

Generate deterministic SVG and PNG figures from the retained validation records:

- **A — Action response**: actual and predicted next-state changes for actions -1, 0, +1, split by model, seed, and coordinate.
- **B — Per-coordinate validation**: position and velocity MAE, contrast error, action order, and clipping by method and seed.
- **C — Representation health**: low-spread dimension count, effective rank, and position/velocity probe R² for each model seed.
- **D — Learning curves**: raw per-coordinate train and validation errors by epoch.
- **E — Gradient magnitudes**: encoder and prediction-head gradient norms for each coordinate and fixed epoch.
- **F — Held-out rollouts**: only if the frozen validation gate opens; position and velocity errors at each preregistered horizon. This is prediction evidence, not planning evidence.

Each figure must state the environment, validation-only status, sample size, metric, and limitations. Include alt text and a caption. Keep all completed and failed seeds visible. No illustrative or fabricated measurements are allowed.
