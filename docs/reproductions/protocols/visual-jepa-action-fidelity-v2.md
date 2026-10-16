# Preregistration: Visual JEPA Action Fidelity v2

**Status**: Frozen before v2 model training  
**Date**: 2026-10-16  
**Study**: [Feature 030](../../../specs/030-jepa-action-fidelity-recovery/spec.md)  
**Report**: [Investigation and results](../visual-jepa-action-fidelity-v2-investigation.md)

## Question and outcomes

**Primary question**: Can a compact action-conditioned visual JEPA learn a non-collapsed representation that retains position, velocity, and counterfactual action effects well enough for stable multi-step predictions?

**Primary validation outcome**: The fraction of paired model/data seeds that pass all representation, probe, one-step state, action-effect, and clipping gates for a fixed objective. No planner outcome is primary.

**Secondary validation outcomes**: Predictive and regularization loss components; encoder and predictor gradient norms and cosine; embedding mean, per-dimension standard deviation, effective rank and norms; probe R², MAE, condition number, coefficient norms and bootstrap stability; latent displacement and action sensitivity; unclipped readout error and clipping frequency; runtime, peak memory, and device.

**Conditional held-out outcomes**: One-step action effects and state errors; latent and state prediction errors at horizons 1, 5, 10, 20, and 60; action-effect ordering; rollout saturation; and fraction of predicted embeddings outside the training distribution.

**Conditional planning outcomes**: Success, return, steps, model transitions, forward batches, planning time, full elapsed time, model seed, and evaluation start. Planning results are not collected unless all preceding gates pass.

## Task and access rules

- Task: synthetic MountainCarContinuous, goal position 0.45, position bounds [-1.2, 0.6], velocity bounds [-0.07, 0.07], action range [-1, 1], and 300-step episode limit.
- Visual input: two consecutive rendered RGB frames resized to 48 × 128. The supervised visual control receives the same frames and action. It predicts next position and velocity directly.
- State-model control: uses simulator position and velocity as input and is marked privileged.
- Exact control: uses the simulator transition function and privileged state.
- Handcrafted visual reference: uses the existing color-center position estimate and two-frame velocity estimate. It is a diagnostic reference, not a learned model.
- No external data, pretrained weights, robot simulator, or additional benchmark is used.
- The direct visual-momentum controller is handcrafted. It is not described as a learned policy.

## Candidate training methods

The three JEPA objectives and supervised visual control use the same 32-dimensional VisualEncoder capacity, frame preprocessing, action input, 35 epochs, batch size 256, AdamW learning rate 0.002, weight decay 0.0001, float32, and paired data/model seeds. The state-space control uses the same transition rows and optimizer budget but receives privileged state. Training uses the first 16 episodes from each 20-episode data group; the last four episodes are validation only.

1. **SIGReg-style JEPA**: existing Epps–Pulley SIGReg implementation; predictive MSE + 0.1 × SIGReg; 128 projections and 17 frequency points. This matches the local v1 objective.
2. **VISReg-style JEPA**: existing local scale, shape, and center regularizer; 0.4 × predictive MSE + 0.6 × regularizer; scale/shape/center weights 1/1/1, 128 projections, epsilon 1e-4. This matches the local Notebook 22 objective.
3. **Variance/covariance JEPA control**: predictive MSE + 0.1 × the existing local variance/covariance proxy with target variance 1, variance weight 1, and covariance weight 0.05. This is a small objective control, not full VICReg.
4. **Supervised visual dynamics control**: same visual input, data, seed pairs, and epoch budget; direct next-state MSE target. It does not use a JEPA predictive objective.
5. **Learned state-space control**: existing state dynamics trained on the same current/next state rows and actions. This control has privileged state input.
6. **Exact and handcrafted controls**: deterministic simulator transitions, and the existing handcrafted visual state estimate followed by exact simulator dynamics. Neither is a learned visual model.

No objective weight, architecture, epoch count, or gate is tuned after reading a held-out record. If a numerical implementation needs a repair before the first fit, update this preregistration and its hash before training; retain this version as the protocol record and document the amendment.

## Seeds and partitions

| Role | Seed values | Use |
|---|---|---|
| Training and validation data | 9301, 9302, 9303 | Twenty random-action episodes per seed; episodes 0–15 train and 16–19 validate. |
| Model initialization | 9401, 9402, 9403 | Pair by index with the three data seeds. Train all candidate objectives and controls. |
| Independent test trajectories | 9501–9505 | Five shared 20-episode groups; 20 predeclared intervention starts at steps 40, 90, 140, and 190. Open only after validation gates pass. |
| Planning starts | 8700–8719 | Reuse the exact v1 final starts only if planning opens. These are paired historical starts, not an independent sample. |
| CPU smoke | 9601–9602 | Pipeline checks only; never used for selection or evaluation. |

Model and data seeds are distinct and paired. Every method uses the same data group for a given pair. Test trajectories, intervention contexts, final planning outcomes, and smoke cases never train model parameters or fit probes.

## Probe and metric rules

- Fit the frozen visual state readout on training embeddings and training state labels only. Select alpha on validation data from {0.01, 0.1, 1.0, 10.0}. Do not allow an unregularized alpha of zero.
- Standardize features with a 1e-3 minimum scale. Solve the ridge system with a stable linear solver. Record the standardized Gram condition number, chosen alpha, raw and standardized coefficient norms, and bootstrap prediction variation.
- Use three seeded episode-block bootstrap refits for validation probe stability. Report the root-mean-square standard deviation of the predicted position and velocity.
- For one-step counterfactuals, hold the same rendered context fixed and evaluate actions -1, 0, and +1. Exact next states come from the simulator dynamics at the corresponding privileged label state.
- Record predicted latent displacement from action 0, finite-difference action sensitivity, predictor output norm, unbounded state readout, clipped state, and saturation indicators.
- Effective rank is exp(entropy of the normalized nonnegative covariance eigenvalues). Report the full spectrum summary and the absolute covariance trace so rank is not read without its scale.
- For rollout out-of-distribution status, compare predicted latents with the training embedding distribution using a training-fit covariance model. Define the threshold before validation: the 99th percentile Mahalanobis distance among training embeddings.
- Exclude exact ties from action-order concordance and report tied and non-tied counts. Never convert a missing or tied result to a correct ranking.

## Frozen validation gates

A JEPA seed passes representation health only when all conditions hold on its validation embeddings:

1. All embeddings and loss outputs are finite.
2. No more than 16 of 32 dimensions have standard deviation below 1e-3.
3. Effective rank is at least 8.
4. The selected probe Gram condition number is at most 1e6, and the three-refit bootstrap prediction RMSE is at most 0.01 position units and 0.005 velocity units.
5. Position probe R² is at least 0.80 and velocity probe R² is at least 0.50; position MAE is at most 0.02 and velocity MAE is at most 0.01.
6. For each coordinate, median normalized error of predicted action contrasts is at most 0.25 across non-tied validation states. The denominator is the true action span with a floor of 1e-4. At least 40 validation states must contribute to each coordinate.
7. Pairwise action-order concordance is at least 0.80 across non-tied position/velocity comparisons.
8. No more than 5% of unbounded next-state predictions fall outside the physical state limits. Always report the unclipped values.

The supervised visual control must pass the one-step MAE, action-contrast, action-order, finite-output, and clipping gates on validation data. If it fails, stop objective-level attribution and investigate the renderer, temporal context, split, and labels. The exact control must match simulator transitions; the state-space control must remain finite and its prediction error must be reported.

For the five-step validation tie-break, use the first available rollout from each of episodes 16–19 in each of the three data groups. Apply the next five recorded actions in that episode. Keep all 12 start rows and mark a missing horizon unsupported.

An objective family is eligible for held-out evaluation only if at least two of its three paired JEPA seeds pass every validation gate. Select at most one eligible objective for held-out evaluation by lowest median normalized action-contrast error, then lowest mean five-step position-plus-velocity MAE on those 12 fixed validation rollouts, then the fixed tie order SIGReg, VISReg-style, variance/covariance. Model selection uses validation data only. Keep all candidate and failed seed rows.

## Held-out and rollout gates

Open the five independent test trajectory groups only after the supervised visual control passes and one JEPA objective is eligible. Evaluate the validation-passing JEPA seeds and matched supervised visual controls on the same 20 test contexts and the same deterministic action sequences. Include exact, state-space, and handcrafted controls. For each context, generate a 60-step continuous action sequence from U[-1, 1] with the existing SHA-256 stable-seed helper, keyed by protocol version, test seed, and start ID. Use the same sequence for every model. Do not tune on the test records.

The selected JEPA must pass the same one-step state and action-effect gates on held-out contexts. For multi-step rollouts, use the same seeded continuous actions in all models and the exact simulator. Per-seed mean absolute state error must be no greater than these fixed limits:

| Horizon | Position MAE | Velocity MAE |
|---:|---:|---:|
| 1 | 0.02 | 0.01 |
| 5 | 0.05 | 0.02 |
| 10 | 0.10 | 0.03 |
| 20 | 0.20 | 0.05 |
| 60 | 0.40 | 0.07 |

At each horizon, report latent prediction error, error growth, clipping, saturation, and out-of-training-distribution fractions. A missing horizon due to environment termination is marked unsupported and is not imputed. Planning remains closed unless at least two validation-passing JEPA model/data pairs pass every supported held-out state/action and rollout gate.

## Conditional planning stage

If the held-out gate passes, use the v1 selected CEM settings without retuning: 60-step horizon, 48 candidates, 8 elites, 4 updates, action blocks of five steps, cumulative Gymnasium return, and both no-prior Gaussian and the fixed visual-momentum prior. Use the same 20 v1 final start seeds (8700–8719). Stop before exceeding the 750,000,000 transition-call ceiling.

Compare exact dynamics, learned state dynamics, supervised visual dynamics, and the validated JEPA. Include direct visual-momentum control. A prior-only arm executes the same fixed visual-momentum rule without learned JEPA dynamics or candidate-model scoring; it is a controller, not an independent planner. Report it once per start. Report JEPA-guided CEM with and without the same prior. A learned visual policy may be added only if the 90 GPU-minute budget leaves enough time; it must use training observations only, be trained before the planning seeds open, and be reported as an extra baseline rather than a matched paper reproduction.

## Failure, abstention, and completeness

- Non-finite values, incomplete seed pairs, or exceeded budgets stop the affected stage. Preserve existing rows and state the exact missing cases.
- A collapsed, ill-conditioned, probe-unstable, action-insensitive, saturated, or rollout-unstable model cannot be presented as a successful JEPA benchmark, even if a planner episode succeeds.
- A poor supervised visual control blocks a JEPA-objective diagnosis until the observation/data path is investigated.
- A failed validation gate closes held-out evaluation. A failed held-out or rollout gate closes planning.
- Do not replace seeds, alter thresholds, change objectives, or retune on test or planning outcomes.
- Negative results with complete evidence are valid. Do not call smoke, validation, or reused v1 start outcomes independent replication.

## Compute, storage, and reproducibility budget

- Maximum training: 15 model fits, 35 epochs each; at most 90 GPU-minutes total on the verified RTX 5070 environment.
- Maximum allocated GPU memory: 2 GiB. Stop if the measured peak exceeds it.
- Maximum transient v2 output on E:: 5 GiB, including checkpoints and arrays. Do not commit datasets, frame tensors, caches, or checkpoints.
- Conditional planning: stop at 750 million model transition calls. No full JEPA-WAM LIBERO-Plus or Push-T run is part of this study.
- Record Python, PyTorch, CUDA, GPU, code hash, protocol hash, data and model seeds, elapsed time, peak allocation, and status for each fit.
- The protocol and implementation hashes must be stored in a lock file before the first model fit. Every later stage validates the lock.

## Figures and reporting

- **A — Action-effect fidelity**: show actual and predicted position and velocity changes for actions -1, 0, +1, by model and seed. Keep unclipped and post-clamp values distinct.
- **B — Representation health**: show per-seed standard deviation, effective rank, position probe R², and velocity probe R².
- **C — Multi-step rollout stability**: show every seed and trajectory at horizons 1, 5, 10, 20, 60; state the aggregate and uncertainty rule.
- **D — Objective comparison**: show validation-only objective validity and action fidelity; identify the selected family using the frozen rule.
- **E — Decision usefulness**: create only if planning opens; show held-out success and return with calls, timing, and the correct start/seed grouping.
- **F — Architecture**: show rendered frames, visual encoder, action-conditioned predictor, predicted latent, frozen state readout, planner, and environment feedback.

Each figure must be regenerated from retained records, use all completed seeds, show sample sizes, include accessible text and a caption, and distinguish historical v1 evidence from v2 measurements. No illustrative measurement is allowed.

## Sources and claim limits

- [LeJEPA](https://arxiv.org/abs/2511.08544) proposes predictive loss with SIGReg. The local model is a small temporal MountainCar adaptation.
- [VISReg](https://arxiv.org/abs/2606.02572) proposes variance scaling and sliced distribution-shape regularization. Notebook 22's local adaptation is not the full paper training recipe.
- [VICReg](https://arxiv.org/abs/2105.04906) motivates variance and covariance terms; the local v2 diagnostic is not a full reproduction.

This study cannot establish a general JEPA advantage, robot transfer, paper reproduction, or independent replication.
