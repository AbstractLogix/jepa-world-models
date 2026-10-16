# JEPA Action Fidelity and Representation Recovery: Investigation

**Evidence version**: v2 investigation, 2026-10-16  
**Starting point**: Draft PR #5 at commit fe57c2e0fdf14ba32b52dd7c12a9289563a8c236  
**Protocol**: [Frozen preregistration v2](protocols/visual-jepa-action-fidelity-v2.md)  
**Feature plan**: [Spec Kit feature 030](../../specs/030-jepa-action-fidelity-recovery/plan.md)

## Question

Can a compact action-conditioned visual JEPA learn a non-collapsed representation that retains position, velocity, and action effects well enough to support stable multi-step predictions?

This work tests why the Notebook 23 visual model did not do so. The goal is an evidence-backed explanation. A negative result is acceptable. It is not a paper reproduction.

## Raw v1 audit

The following counts and values were recomputed from the retained v1 CSV files. No v1 file was changed.

| Check | Raw result |
|---|---|
| Completed model fits | 15/15: five state models, five image models, five JEPA models |
| JEPA embedding spread | All five have 32/32 dimensions below 1e-3; mean per-dimension standard deviation ranges from 0.0000142 to 0.0001072 |
| JEPA position probe | R² ranges from 0.972912 to 0.996362 across five test records |
| JEPA velocity probe | R² ranges from -0.011876 to 0.095020 across five test records |
| JEPA action triplets | 20 starts per seed, paired with actions -1, 0, +1 |
| Predicted action span | Seeds 7402, 7404, and 7405 have zero predicted position span on all 20 starts. Seeds 7401 and 7403 span the full 1.8 position interval after clamping. |
| Actual action span | Median one-step span is about 0.003 in both position and velocity across the 20 starts per seed. |
| Clamped predictions | The intervention CSV contains positions at -1.2 or +0.6 and velocities at -0.07 or +0.07. The raw CSV stores post-clamp states, so it cannot establish the pre-clamp clipping rate. |
| JEPA CEM | 0/100 successes with no-prior Gaussian initialization and 0/100 with the momentum prior |
| Exact-dynamics CEM | 0/20 without the momentum prior; 20/20 with it |
| Learned state-space CEM | 0/100 without the prior; 99/100 with it |
| Completed episode records | 660/660; the separate v1 validator also records 200 development rows, 15 model fits, 10 probe rows, and 960 interventions |

The exact and state-model outcomes confirm that the locked CEM can solve this task under the momentum prior when it has adequate dynamics. They do not establish that a visual model can do so.

Notebook 22 is separate historical evidence. It uses a VISReg-style temporal-prediction adaptation, not the v1 LeJEPA-style SIGReg setting. Its final JEPA fits had standard deviations of 0.00279–0.00574, 153/295 non-tied action-pair rankings correct, 0/100 uniform-start successes, and 100/100 momentum-prior successes. Direct visual-momentum control reached 20/20. The different objective and planner outcomes are not a controlled objective comparison.

## V1 checkpoint replay

All five retained v1 checkpoints were replayed with the saved probe and fixed intervention contexts. The replay matched all 300 saved post-clamp position and velocity predictions exactly. It did not edit the v1 checkpoints, notebooks, protocol, or evidence.

| Diagnostic | Replayed result |
|---|---|
| Context and target spread | Context mean standard deviation 0.0000147–0.0000882; target mean standard deviation 0.0000147–0.0000889. All 32 dimensions in each distribution are below the v1 1e-3 rule. |
| Effective rank | Context rank 1.36–3.99; target rank 1.38–4.00. |
| Encoder and predictor norms | Context encoder output norm is 5.65685 on these rows, consistent with the per-sample LayerNorm scale. Total predictor norm across each three-action batch is 9.63–9.85. Norm alone does not show between-sample health. |
| Probe conditioning | V1 minimum feature scale is 1e-4 for every seed. Two seeds selected alpha 0. The regularized Gram condition number ranges from 95,012 to 31,923,390; raw probe coefficient L2 norm ranges from 8,234 to 28,583. |
| Unclipped state readout | All 300 predictions change under task-bound clipping. Raw position spans -74.13 to 780.74; raw velocity spans -646.51 to 29.30. |
| Action effect after decoding | Median absolute raw position change between action -1 or +1 and action 0 is 5.11–92.65 across seeds; the corresponding actual median change is about 0.0015. The velocity change is 1.62–27.79 against about 0.0015 actual. Action ordering is inconsistent across seeds and coordinates. |
| Probe bootstrap | Across 20 episode-block refits per seed, position prediction variation RMSE is 0.0014–0.0067 and velocity is 0.0005–0.0156. Seed 7404 has the largest velocity variation. |
| Saved-checkpoint gradients | On one reconstructed training batch per checkpoint, the weighted encoder SIGReg gradient is about 135–370 times the predictive encoder gradient. Cosine similarity ranges from -0.44 to 0.16. The regularizer gradient is not near zero on these batches. |

The replay confirms a failure path: a very low-spread representation is followed by a large, poorly conditioned linear readout, and predictor outputs decode far outside the physical state range. This explains why some post-clamp seeds appear action-insensitive while others span the full allowed position range. It does not isolate whether the main cause is the collapsed latent geometry, the probe floor and unregularized v1 alpha choices, predictor outputs leaving the probe's training distribution, or their interaction. V1 did not save enough latent vectors to measure its predictor outputs against a training-distribution distance reference. V2 will record this directly if its validation gates pass.

## V2 validation results

The v2 protocol and implementation hashes were locked before training. The bounded CUDA validation completed all 15 preregistered fits: three seeds for each of three JEPA objectives, the supervised visual control, and the privileged-state model. All fits completed. Training used 14.28 GPU-minutes on an NVIDIA RTX 5070, with 482.33 MiB maximum allocated GPU memory. The independent test split remained closed because the supervised visual control failed its frozen gate. No held-out rollout, CEM planning, or planning outcome was collected.

### Observation and label controls

The observation audit regenerated the three paired data groups without fitting a model. Each group contains 4,800 training and 1,200 validation transitions. Across both splits, the fixed two-frame color-center estimator had position MAE 0.003747–0.003751 and velocity MAE 0.0000555–0.0000574. Its 95th-percentile absolute errors were at most 0.003834 position and 0.000138 velocity. The simulator transition function matched stored next-state labels to a maximum absolute error below 6e-8 across the 18,000 regenerated transitions. This confirms that the renderer, temporal context, split, and labels contain enough information for this task. It does not show that a learned visual model can recover that information.

On the 20 preregistered action contexts per data seed, the handcrafted visual estimate followed by exact dynamics produced position MAE 0.00366–0.00375 and velocity MAE 0.000045–0.000080. Its normalized position and velocity action-contrast errors were below 1e-5 and 5e-8, with correct action ordering on all 20 starts in each group. The exact privileged-state simulator also matched all tested transitions. The learned state-space control remained finite and had position MAE 0.00046–0.00163 and velocity MAE 0.00092–0.00214; it is a privileged-state reference.

The supervised visual dynamics model failed validation on all three model/data pairs. Its position MAE was 0.0261–0.0803, above the frozen 0.02 limit, while velocity MAE was 0.00424–0.00704. Normalized action-contrast error was 0.236–1.415 for position and 0.573–0.733 for velocity, against the 0.25 limit. The three final validation prediction losses were 0.000438–0.006244, but a low average MSE did not satisfy the state or action-fidelity gates. This failure stops objective-level attribution under the preregistered rule.

### Objective and representation findings

| Objective | Context and target spread | Effective rank | Position probe R² | Velocity probe R² | Position MAE | Action-contrast error, position / velocity | Result |
|---|---|---:|---:|---:|---:|---:|---|
| SIGReg-style | All 32 dimensions below 1e-3 in all three seeds | 1.77–4.33 | 0.249–0.947 | -0.0015–0.0017 | 17.0–26.2 | 69.3–178.0 / 0.34–1.20 | 0/3 pass; all 180 action predictions clip |
| VISReg-style | No dimensions below 1e-3 in all three seeds | 9.44–12.09 | 0.970–0.985 | 0.0082–0.0180 | 0.246–5.03 | 5.60–40.25 / 0.31–6.58 | 0/3 pass; one seed clips all 60 action predictions |
| Variance/covariance control | All 32 dimensions below 1e-3 in all three seeds | 6.14–7.58 | 0.899–0.946 | 0.0066–0.0140 | 0.096–0.201 | 25.6–126.5 / 0.58–4.21 | 0/3 pass; rank below 8 in every seed |

All nine JEPA fits failed the validation gate, and no objective was selected. VISReg-style training cleared the numeric spread and rank screens, but it did not produce useful velocity probes or action-conditioned state predictions. Thus non-collapsed spread alone was not enough. Position probes were often high, while velocity probes remained close to zero for every objective. The three objective settings are local adaptations, not the complete recipes or benchmarks from the cited papers.

The action predictor was not uniformly insensitive at the latent level: median latent displacement from the zero-action prediction ranged from 0.0086 to 0.0692. However, the resulting state readout had incorrect action magnitudes and, in some seeds, incorrect direction. The actual one-step action change at the sampled contexts was small (median about 0.0015 state units), while decoded JEPA contrasts were much larger. SIGReg raw position predictions ranged from -26.9 to -17.2 across its three seeds; VISReg-style predictions ranged from -0.90 to 4.70. The validation probe's selected alpha was positive and its Gram condition number was 1.3–512.5, so the v2 failures are not explained by the extreme v1 Gram conditioning alone. V2 raw probe coefficient norms were 11.6–392.6.

The encoder output norm was about 5.6568 for each visual context, as expected from the encoder's final per-sample normalization. This norm does not measure between-sample representation spread. The predictor output norm was about 9.57–12.23, and predicted latent norms were about 5.53–7.06. These finite norms do not make the action readout valid.

Gradient snapshots were recorded at epochs 1, 5, 10, 15, 20, 25, 30, and 35. The diagnostic computes gradients of the unweighted prediction and regularizer components separately. After applying each objective's frozen regularizer weight, the median SIGReg encoder-regularizer gradient was about 133–157 times its prediction gradient across the three seeds. The VISReg-style weighted ratio was about 1.5–5.5, and the variance/covariance weighted ratio was below 0.001. This gradient imbalance is measured, but it is not a causal ablation and does not prove why SIGReg collapsed. The protocol did not permit post-result weight tuning.

### What the evidence does and does not identify

Confirmed: v1 has near-constant embeddings, unstable probe coefficients, and extreme pre-clamp state outputs. In v2, SIGReg and the variance/covariance control again have low spread; VISReg-style has adequate spread but weak velocity information; all three objectives fail action fidelity; and the supervised visual model fails despite a verified visual state signal and correct labels. The action-conditioning, representation, and state readout do not jointly support reliable one-step dynamics under this protocol.

Not identified: whether SIGReg's gradient imbalance caused its collapse; whether the weak velocity signal comes from the loss, optimizer, architecture, or one-step target scale; whether predicted latents are out of the training distribution; and whether errors accumulate during autoregressive planning. The last two questions require a valid model and an opened rollout stage. Because the validation gate failed, the independent test data and planner were not accessed. This is an intentional stop, not a missing successful run.

The frozen v2 planner design labels its direct visual-momentum controller as the prior-only arm. It does not define a separate prior-only CEM planner without learned JEPA dynamics, although the requested future comparison distinguishes both. No planning arm ran because validation failed. A future planning study must add that separate planner in a new preregistration amendment and lock its implementation hash before opening the stage.

Retained records include the validation fit, training-history, gradient, action-triplet and summary tables; the closed held-out summary; the replayed v1 checkpoint diagnostics; and a separate observation-control audit. Their file hashes are recorded in [the v2 evidence manifests](../evidence/visual-jepa-action-fidelity-v2-evidence-manifest.json) and [the observation audit manifest](../evidence/visual-jepa-action-fidelity-v2-observation-control-manifest.json). CUDA checkpoints and full transient outputs remain on E:.

## Root-cause plan recorded before model changes

The Notebook 22 and 23 implementations and all v1 evidence remain unchanged. At preregistration freeze, the separate v2 models had not been fit. The following diagnostic order was fixed before model changes:

1. Replay the v1 JEPA checkpoints without changing them. Measure encoder norms, latent covariance spectrum and effective rank, predictor output norms, action derivatives, action-triplet latent displacement, and decoded states before and after task-bound clipping.
2. Recompute probe design conditioning and coefficient magnitudes. Compare ridge selection, feature scaling, train-only fit, validation selection, and bootstrap prediction stability. Report whether the readout, rather than the predictor, creates the boundary-saturated state output.
3. Measure prediction, SIGReg, and total gradient norms separately for encoder and predictor on fixed train batches. Include gradient cosine and exact-zero / near-constant synthetic checks. A poor outcome alone does not establish a loss defect.
4. Run positive controls on matched visual inputs: exact MountainCar transitions, learned state-space dynamics with privileged state, the fixed color-center visual estimator, and a supervised visual dynamics model with the same rendered frames, actions, split, architecture scale, and training budget.
5. Compare the frozen SIGReg, VISReg-style, and variance/covariance objective settings on three paired model/data seeds. Keep the same image encoder, action predictor, data, batch size, optimizer, precision, and epoch count. Record the loss components and diagnostics separately.
6. Open independent test trajectories only if the supervised visual control and selected JEPA objective meet all validation gates. Open the planner only if held-out one-step action fidelity and multi-step stability also pass.

## Confirmed implementation facts

- Notebook 23 sends two consecutive 48 × 128 RGB frames to a 32-dimensional convolutional encoder. The action predictor concatenates a scalar action to that embedding and predicts a target embedding.
- The JEPA uses a shared online encoder for context and target. This is allowed by the LeJEPA objective family; the absence of a stop-gradient is not, by itself, evidence of a defect.
- The v1 regularizer is applied to concatenated current and target embeddings. The local loss is predictive MSE plus 0.1 times the local Epps–Pulley SIGReg term, with 128 projections and 17 frequencies.
- V1 probe features are scaled with a 1e-4 floor. The alpha search includes zero and fits from a Gram matrix. The replay confirms large raw coefficient norms, high condition numbers for four of five seeds, and extreme unbounded readouts. Readout amplification is part of the failure, but its share relative to predictor distribution shift remains open.
- The planner clips decoded position and velocity to task bounds after each latent prediction. The saved intervention states are therefore not raw probe outputs.
- V1 did not include a supervised visual state-prediction control, a frozen action-fidelity gate, effective rank, gradient-by-loss records, unclipped readouts, or autoregressive rollout diagnostics.

## Hypotheses, not findings

| Hypothesis | Evidence needed before it can be stated as a cause |
|---|---|
| SIGReg's regularizer interaction causes predictive information loss. | V1 checkpoint replay and v2 training snapshots show a large regularizer-to-prediction encoder-gradient ratio. The v2 ratio is not a causal ablation, so the loss interaction is confirmed but its causal role remains open. |
| The visual context contains too little or too noisy velocity information. | The handcrafted estimator recovered position and velocity across all train and validation episodes. This rules out missing visual information as the cause in this renderer. The learned visual control still failed; its learning cause is unresolved. |
| The predictor ignores action in some seeds and overreacts in others. | Latent action displacement is nonzero in v1 and v2, but decoded effects are far too large or misdirected. The latent distribution of predicted outputs was not assessed against the training reference because no rollout gate opened. |
| Probe scaling amplifies small residual variation or extrapolates poorly. | V1 readout amplification is confirmed by large coefficient norms and out-of-range decoded predictions. V2 uses positive-alpha ridge probes with moderate Gram condition numbers, yet some predictions still leave physical bounds. Probe instability alone does not explain the v2 failures. |
| A valid one-step representation becomes unstable during recursive planning. | Not measured. The v2 model failed validation, so test rollouts, saturation accumulation, and planning OOD distances were not run. |
| The one-step supervised loss underweights small action effects. | The visual model's average validation MSE is low for two seeds, but action-contrast checks fail. This is consistent with a low-signal action effect, but no loss-scale or horizon ablation was run and the cause is not confirmed. |

The v1 results establish near-collapse and poor state/action decoding for that configuration. They do not yet identify which of these mechanisms caused the failure.

## Objective research boundary

LeJEPA combines predictive loss with SIGReg. VISReg uses a variance scale term and a sliced-Wasserstein distribution-shape term. Notebook 22 and Notebook 23 use local temporal-prediction adaptations rather than the complete paper training recipes. The v2 comparison checks whether the loss behavior changes this compact MountainCar model; it does not test the papers' broader claims. The simple variance/covariance track is a local diagnostic control, not a full VICReg reproduction.

Primary sources: [LeJEPA](https://arxiv.org/abs/2511.08544), [VISReg](https://arxiv.org/abs/2606.02572), and [VICReg](https://arxiv.org/abs/2105.04906).

## Evidence status

The v1 raw audit and checkpoint replay and the v2 validation study are complete. The v2 protocol and two pre-training amendments were frozen and locked before fitting. The validation gate failed, no objective was selected, and independent test trajectories, multi-step rollouts, and planning remain closed. The validation-only result is negative and exploratory; it is not a paper reproduction or a general JEPA conclusion.
