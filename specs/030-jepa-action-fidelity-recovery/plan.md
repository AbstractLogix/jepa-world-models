# Implementation Plan: JEPA Action Fidelity and Representation Recovery

**Branch**: codex/jepa-action-fidelity-recovery | **Date**: 2026-10-16 | **Spec**: [spec.md](spec.md)

## Summary

Audit the raw Notebook 23 evidence, freeze a v2 validation protocol, then build a separate diagnostic runner and controls. Do not change v1 code or evidence. The first implementation stage is diagnostic, not model tuning. Run held-out and planning stages only when the frozen gates pass.

## Technical Context

**Language/Version**: Python 3.10 through 3.13; bounded CUDA runtime verified on the E: volume with Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 5070.

**Primary Dependencies**: Existing PyTorch, NumPy, Gymnasium, Pillow, Matplotlib, and standard-library tools. No new package is planned.

**Storage**: Keep run outputs, checkpoints, and temporary arrays on E:. Commit only compact CSV/JSON records and figures.

**Testing**: Existing unittest suite, targeted v2 tests, Ruff, package build, evidence validator, CPU smoke, and deterministic figure regeneration.

**Target Platform**: Linux/WSL development with the verified E:-hosted CUDA environment; CPU remains the contributor smoke path.

**Project Type**: Python research tutorial and reproducibility tools.

**Performance Goals**: Validation training within 90 GPU-minutes; peak allocated GPU memory below 2 GiB; at most 5 GiB of transient storage.

**Constraints**: No full JEPA-WAM benchmark, Push-T evaluation, new dataset, or unrelated architecture. Planning is conditional and capped at the existing v1 transition budget.

**Scale/Scope**: Three paired model/data seeds; 35 epochs; three JEPA objective variants, one supervised visual control, one privileged state-space control, exact dynamics, and a handcrafted visual estimator.

## Constitution Check

- Teach from evidence: PASS. Every result uses retained raw records and a versioned protocol.
- Label examples honestly: PASS. JEPA adaptations, controls, and synthetic results stay distinct.
- Usable learning path: PASS. Add CPU smoke and clear E: output instructions.
- Measure before claims: PASS. Freeze gates and seeds before training.
- Small and reviewable: PASS. Use a separate v2 runner and records; do not refactor or rewrite v1.
- Preserve licensing and data limits: PASS. No external asset or checkpoint is added.

## Root-Cause Investigation Plan

This plan is recorded before changes to model code. The supporting audit is [visual-jepa-action-fidelity-v2-investigation.md](../../docs/reproductions/visual-jepa-action-fidelity-v2-investigation.md); the frozen gates are in [protocol v2](../../docs/reproductions/protocols/visual-jepa-action-fidelity-v2.md).

### Confirmed v1 evidence

The raw v1 records contain 15 model fits, 10 held-out probe rows, 960 action-intervention rows, and 660 complete episode rows. All five JEPA fits (model seeds 7401–7405) are finite but have all 32 embedding dimensions below the recorded 1e-3 spread threshold. Their mean per-dimension standard deviation is 1.42e-5 to 1.07e-4. Held-out position-probe R² is 0.9729–0.9964; velocity-probe R² is -0.0119–0.0950.

Each JEPA seed has 20 action-triplet starts. Seeds 7402, 7404, and 7405 predict zero position span across actions. Seeds 7401 and 7403 reach the full 1.8 position span after the state readout is clipped to [-1.2, 0.6]. Raw actual median action spans are about 0.003 for both position and velocity. The clipped predictions contain values at physical bounds. The v1 CSVs do not retain the unclipped state readout, so the pre-clip saturation frequency is unknown.

JEPA CEM succeeds on 0/100 episodes with either no-prior Gaussian initialization or the momentum prior. Exact dynamics succeed on 20/20 with the prior and 0/20 without it. Learned privileged state dynamics succeed on 99/100 with the prior and 0/100 without it. This confirms that the locked planner can solve this task with the prior and a suitable dynamics model; it does not establish visual-JEPA quality.

Notebook 22 used a VISReg-style temporal-prediction adaptation. Its five final JEPA seeds cleared the 1e-3 low-spread screen, but had low overall spread (0.00279–0.00574), 153/295 non-tied latent action pairs ranked correctly, 0/100 uniform-start CEM successes, and 100/100 momentum-prior CEM successes. The direct hand-coded visual-momentum controller succeeded on 20/20. This does not establish that VISReg improves action fidelity.

### Hypotheses and tests

| Candidate cause | Current status | Test and evidence needed |
|---|---|---|
| Encoder collapse or near-collapse | Confirmed for all five v1 JEPA representations by the recorded spread rule. The cause is not known. | Replay v1 checkpoints; compute variance, covariance spectrum, effective rank, output norms, and finite status. Compare v2 objectives on validation data. |
| Velocity is not recoverable from the two rendered frames | Open hypothesis. The renderer shows position only, but the context contains consecutive frames and Notebook 22's handcrafted estimator uses their motion. | Evaluate exact-state and handcrafted visual controls; train a supervised visual dynamics model on the same frames, actions, and split. |
| Predictor ignores or overreacts to action | Open hypothesis. The post-readout, post-clamp v1 action spans include both zero and full-range outputs. | Measure raw latent deltas and d(predicted latent)/d(action); compare unclipped state contrasts with exact simulator contrasts on identical contexts. |
| SIGReg gradients or predictive/regularization interaction drive collapse | Open hypothesis. A prior local gradient check found zero SIGReg input gradient at an exact zero embedding, but did not establish a v1 training defect. | Measure predictive and regularizer gradient norms and cosine by module at initialization and fixed training checkpoints; inspect v1 checkpoint gradients and compare fixed v2 objectives. |
| Probe scaling amplifies tiny latent differences | Implementation risk, causal effect unconfirmed. V1 allows alpha=0 and floors feature scale at 1e-4; selected alpha, design conditioning, and coefficient norms vary by seed. | Record train-feature condition numbers, selected alpha, raw and standardized coefficient norms, bootstrap probe stability, and pre/post-clamp readouts. |
| Latent normalization is unsuitable | Open hypothesis. The encoder ends in non-affine LayerNorm, but per-sample LayerNorm does not guarantee between-sample spread. | Compare embedding and predictor norms, per-dimension spread, effective rank, and component gradients across fixed objectives. Do not change normalization until these records support it. |
| Autoregressive errors or planner-distribution shift cause failure | Open hypothesis. V1 planning repeatedly feeds predicted latents to the predictor and clips the decoded state; no v1 rollout diagnostics record the unbounded state or train-distribution distance. | After validation gates pass, compare horizons 1, 5, 10, 20, and 60; log state error, latent error, clipping, saturation, and distance from training latents. |

## Design

1. Preserve v1 paths and evidence. Give every new source, protocol, record, and figure a v2 name.
2. Build a validation-only runner that regenerates deterministic MountainCar data, applies the episode split, trains matched objective/control models, and writes fail-closed per-seed rows.
3. Keep the three JEPA candidate objectives on the same 32-dimensional visual encoder, action predictor, optimizer, training data, epochs, and seed pairs. Keep the supervised visual model's visual encoder and action input comparable while predicting simulator state directly.
4. Measure action triplets from one held-out visual context. Keep privileged state only in the exact and state-space controls and in the environment-side labels.
5. Fit robust probes on train data, select them on validation data, and record conditioning and coefficient scales. Never fit with held-out labels.
6. Open held-out rollouts only after the supervised visual control and selected JEPA family pass validation. Open planning only after the selected JEPA also passes independent action and rollout gates.
7. Generate separate figures from versioned evidence. Do not add synthetic or illustrative rows.
8. Update the roadmap, evaluation protocol, reproduction matrix, technical report, case study, and JSON manifest after evidence status is known.

## Project Structure

```text
src/jepa_tutorial/action_conditioned_visual.py
src/jepa_tutorial/action_fidelity_v2.py
scripts/run_visual_jepa_action_fidelity_v2.py
scripts/analyze_visual_jepa_action_fidelity_v2.py
scripts/validate_visual_jepa_action_fidelity_v2.py
tests/test_visual_jepa_action_fidelity_v2.py
docs/reproductions/protocols/visual-jepa-action-fidelity-v2.md
docs/reproductions/visual-jepa-action-fidelity-v2-investigation.md
docs/evidence/visual-jepa-action-fidelity-v2-*.csv
docs/evidence/visual-jepa-action-fidelity-v2-*.json
docs/portfolio/figures/visual-jepa-action-fidelity-v2/*.{svg,png}
```

**Structure Decision**: Keep the new experiment in a separate v2 module and runner. Use the existing Notebook 23 only as the frozen starting evidence. Do not edit Notebook 22 or 23 or reuse their output paths. This milestone does not add a new lesson notebook.

## Validation

- Unit tests cover action sensitivity, constant and non-finite embeddings, probe conditioning, clipping, rollouts, deterministic seeds, split isolation, evidence completeness, file integrity, and figure regeneration.
- CPU smoke writes only reserved smoke outputs and never opens test or planning cases.
- Run the bounded objective/control fit on CUDA only after v2 code and protocol hashes are frozen.
- Run the full existing unit suite, Ruff, package build, evidence validation, CPU smoke, and deterministic figure checks before publication.

## Risks and Limits

- A low loss or spread pass does not imply action fidelity or decision value.
- The visual positive control may fail because the renderer's motion signal is quantized. That outcome blocks an objective-level conclusion.
- Three paired seeds support a bounded validation study, not broad statistical claims.
- V2 planning starts reuse the v1 final seeds only if the planning gate opens. They are paired with the historical protocol and are not an independent task sample.
- LeJEPA and VISReg use different regularization ideas; the local temporal prediction implementations are not full reproductions. LeJEPA combines prediction and SIGReg; VISReg replaces covariance with a sliced-Wasserstein shape term while keeping variance scale control. The study tests local adaptations only. See the primary [LeJEPA](https://arxiv.org/abs/2511.08544) and [VISReg](https://arxiv.org/abs/2606.02572) papers.
