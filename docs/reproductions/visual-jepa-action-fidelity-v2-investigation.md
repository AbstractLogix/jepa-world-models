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

## Root-cause plan recorded before model changes

The Notebook 22 and 23 implementations and all v1 evidence remain unchanged. The separate v2 models have not been fit. The following diagnostic order is fixed:

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
- V1 probe features are scaled with a 1e-4 floor. The alpha search includes zero and fits from a Gram matrix. Those choices can make a probe numerically sensitive; their effect on the saved predictions remains to be measured.
- The planner clips decoded position and velocity to task bounds after each latent prediction. The saved intervention states are therefore not raw probe outputs.
- V1 did not include a supervised visual state-prediction control, a frozen action-fidelity gate, effective rank, gradient-by-loss records, unclipped readouts, or autoregressive rollout diagnostics.

## Hypotheses, not findings

| Hypothesis | Evidence needed before it can be stated as a cause |
|---|---|
| The SIGReg gradient vanishes near the v1 representation and lets prediction collapse dominate. | Per-component encoder and predictor gradient norms at saved checkpoints and throughout fixed v2 training batches; compare against VISReg-style and variance/covariance losses. |
| The visual context contains too little or too noisy velocity information. | Handcrafted visual estimate and matched supervised visual dynamics results on held-out validation and test trajectories. |
| The predictor ignores action in some seeds and overreacts in others. | Raw predicted latent changes and unbounded decoded state differences for the same observation under three actions. |
| Probe scaling amplifies small residual variation. | Train design condition numbers, coefficient norms, bootstrap probe stability, and direct comparison of unbounded readout with clipped intervention rows. |
| A valid one-step representation becomes unstable during recursive planning. | Held-out state and latent errors at horizons 1, 5, 10, 20, and 60, together with saturation and training-distribution distance. |

The v1 results establish near-collapse and poor state/action decoding for that configuration. They do not yet identify which of these mechanisms caused the failure.

## Objective research boundary

LeJEPA combines predictive loss with SIGReg. VISReg uses a variance scale term and a sliced-Wasserstein distribution-shape term. Notebook 22 and Notebook 23 use local temporal-prediction adaptations rather than the complete paper training recipes. The v2 comparison checks whether the loss behavior changes this compact MountainCar model; it does not test the papers' broader claims. The simple variance/covariance track is a local diagnostic control, not a full VICReg reproduction.

Primary sources: [LeJEPA](https://arxiv.org/abs/2511.08544), [VISReg](https://arxiv.org/abs/2606.02572), and [VICReg](https://arxiv.org/abs/2105.04906).

## Evidence status

At this point, only the v1 audit is complete. The v2 protocol is frozen separately. V2 training, held-out trajectories, multi-step tests, and planning have not yet run. No new objective has been selected from held-out outcomes.
