# Visual JEPA action-fidelity v2: technical report

**Status:** Validation complete; held-out and planning gates closed

**Protocol:** [Frozen v2 preregistration](reproductions/protocols/visual-jepa-action-fidelity-v2.md)
**Full investigation:** [Root-cause report](reproductions/visual-jepa-action-fidelity-v2-investigation.md)

## Question

Can a compact action-conditioned visual JEPA learn a non-collapsed representation that retains position, velocity, and action effects well enough to support multi-step planning?

## Result

Not under the frozen v2 training recipe. All nine JEPA fits failed at least one validation gate. The supervised visual dynamics control also failed, so the preregistered rule stopped objective attribution and kept independent test data and planning closed. This is a local negative result for the tested synthetic MountainCar setup. It is not a general JEPA result or a reproduction of LeJEPA or VISReg.

## Design

The protocol used rendered two-frame RGB contexts, scalar actions, 20 synthetic episodes per data seed, three paired model/data seeds, and a 35-epoch training budget. Episodes 0–15 trained each model; episodes 16–19 were validation only. All visual methods used the same 32-dimensional encoder and paired data. The candidate methods were:

- SIGReg-style action-conditioned JEPA.
- VISReg-style action-conditioned JEPA.
- A local variance/covariance JEPA control.
- Supervised visual next-state dynamics.
- A privileged learned state-space model.
- Exact simulator and handcrafted visual-state controls.

The v2 protocol fixed representation, probe, one-step state, action-fidelity, clipping, compute, and storage gates before fitting. The supervised visual control had to pass before any objective could be selected. The models and protocols from Notebooks 22 and 23 were not changed.

## Verified findings

1. **The rendered inputs and labels carry the required state information.** On 18,000 deterministically regenerated transitions, the handcrafted two-frame estimate had 0.00375 position MAE and 0.000056 velocity MAE on both the training and validation partitions. The exact simulator matched the stored next-state labels within 6×10⁻⁸. This rules out missing position or velocity information in this renderer as the reason for the learned visual control failure.

2. **A non-collapsed representation did not retain useful velocity or action information.** All three VISReg-style seeds passed the frozen spread and rank screens: mean embedding standard deviation was 0.00357–0.01217 and effective rank was 9.44–12.09. Their position-probe R² was 0.970–0.985, but velocity-probe R² was only 0.0082–0.0180. None passed action fidelity.

3. **SIGReg and variance/covariance again had low spread.** All 32 dimensions fell below the 1e-3 threshold for all six seeds in these two objective groups. Effective rank was 1.77–4.33 for SIGReg and 6.14–7.58 for variance/covariance, below the required rank of 8.

4. **The supervised visual control did not pass.** Its three position MAEs were 0.0261, 0.0343, and 0.0803 against a 0.02 limit. Velocity MAE was 0.00424–0.00704, but action-contrast error exceeded the 0.25 limit for every seed in at least one state coordinate. A low average one-step MSE did not establish faithful action effects.

5. **The decoder and action response remained unreliable.** The predictor produced nonzero latent changes for changed actions, but the frozen state readout mapped them to action effects much larger than the simulator's median one-step effect of about 0.0015 state units. SIGReg predictions clipped on all 180 action rows. One VISReg-style seed clipped on all 60 rows. Some unclipped probes had moderate Gram condition numbers, so the v1 conditioning failure is not the only observed problem.

6. **SIGReg gradients were strongly imbalanced in the local implementation.** At eight snapshots per seed, after applying the frozen 0.1 regularizer weight, the median encoder regularizer-gradient norm was about 133–157 times the prediction-gradient norm. This is a measured loss interaction, not a causal test of why the representation collapsed.

## Compute and evidence

The study completed 15 of 15 fits on an NVIDIA RTX 5070. Training used 14.28 GPU-minutes; maximum allocated GPU memory was 482.33 MiB. All transient datasets and checkpoints remain on E:. The repository retains compact validation records, an evidence hash manifest, the closed-stage summary, and deterministic figures. The test split, autoregressive rollout stage, and planner were not opened.

The figures separate historical v1 replay from v2 validation evidence. Figure C (multi-step rollout) and Figure E (decision outcome) were not generated because their gates stayed closed. No held-out success, return, rollout, or planning value is available for v2.

## Root-cause assessment

**Confirmed:** v1 had near-constant embeddings, high probe coefficient magnitudes, and extreme raw state predictions that saturated after clipping. In v2, the SIGReg and variance/covariance settings again had low spread. VISReg-style spread improved, but velocity probes remained weak. All JEPA objectives failed action-fidelity checks. The supervised visual dynamics model failed despite clean observation and label controls.

**Still open:** the causal role of the SIGReg gradient imbalance; whether one-step state loss underweights small action effects; whether predictor outputs leave the training latent distribution; whether rollout errors grow with horizon; and whether any valid representation can improve decisions over a direct policy. These require a new preregistered study. They must not be answered by tuning against this validation result.

The frozen v2 planning design treats its prior-only arm as the direct visual-momentum controller. It does not include a separate CEM planner with that prior but without learned JEPA dynamics. The planning gate stayed closed, so no arm was run. Any future planning protocol must add that separate prior-only planner and freeze a new protocol and code hash before planning opens.

## Reproduction

From the repository root, with the locked CUDA environment and all transient files directed to E: or D::

```bash
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py verify-lock
uv run --locked --no-sync python scripts/audit_visual_jepa_action_fidelity_v2_observations.py
uv run --locked --no-sync python scripts/validate_visual_jepa_action_fidelity_v2.py --run-dir "$JEPA_V2_OUTPUT_DIR/validation" --v1-audit-dir "$JEPA_V2_OUTPUT_DIR/v1-audit"
uv run --locked --no-sync python scripts/validate_visual_jepa_action_fidelity_v2.py --check-integrity
uv run --locked --no-sync python scripts/audit_visual_jepa_action_fidelity_v2_observations.py --check-integrity
uv run --locked --no-sync python scripts/analyze_visual_jepa_action_fidelity_v2.py --output-dir docs/portfolio/figures/visual-jepa-action-fidelity-v2
uv run --locked --no-sync python scripts/analyze_visual_jepa_action_fidelity_v2.py --output-dir docs/portfolio/figures/visual-jepa-action-fidelity-v2 --verify-determinism
```

The final validation used the locked settings of 35 epochs and batch size 256. The preregistration caps training at 90 GPU-minutes, allocated memory at 2 GiB, and transient storage at 5 GiB. Re-running the fits is unnecessary to regenerate the retained figures.

## Next experiment

Do not tune this failed validation set. A follow-up should first preregister a visual positive-control intervention that targets small action effects and uses the same frame pairs, training episodes, and validation episodes. It should report per-coordinate state loss, action-contrast error, and seed-level gradients. Open a fresh independent test split only after that control passes. Then test one selected JEPA configuration on held-out action interventions and multi-step rollouts before considering planning. A planning amendment must add a prior-only planner separately from direct visual-momentum control and JEPA-guided planning with and without the same prior.

See the [roadmap](ROADMAP.md), [reproduction matrix](REPRODUCTION_MATRIX.md), and [portfolio case study](portfolio/CASE_STUDY.md) for the project-level status.
