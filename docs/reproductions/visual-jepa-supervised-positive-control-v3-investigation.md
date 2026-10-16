# Supervised Visual Positive-Control v3: Investigation Report

**Status:** Incomplete. The validation run stopped before an optimizer update.

**Date:** 2026-10-16

**Protocol:** [Frozen v3 preregistration](protocols/visual-jepa-supervised-positive-control-v3.md)

**Feature:** [Spec Kit 031](../../specs/031-supervised-visual-positive-control-v3/spec.md)

**Run record:** [summary](../evidence/visual-jepa-supervised-positive-control-v3/run-summary.json) · [evidence manifest](../evidence/visual-jepa-supervised-positive-control-v3/sha256-manifest.json)

## Executive result

The v3 data path passed all three validation groups. The two-frame observations recover position and velocity with the fixed color-center estimator. The exact simulator matches stored labels. The fixed handcrafted-feature ridge control runs, but its one-step errors are much larger than the exact dynamics control.

The first neural fit stopped during `loss.backward()`. PyTorch 2.10.0+cu128 reported that `adaptive_avg_pool2d_backward_cuda` has no deterministic implementation under the frozen `torch.use_deterministic_algorithms(True)` setting. The optimizer step follows the backward call, so no optimizer update completed. The run records 0/15 completed neural fits, one failed first fit, and 14 later fits as `not_run`.

This is an execution failure for the v3 neural study. It is not evidence that the learned state estimator or visual dynamics model fails scientifically. The validation gate is incomplete. The independent test set, multi-step rollouts, and planning remain closed. We did not change devices or deterministic settings after the run.

## PR #6 review

[PR #6](https://github.com/AbstractLogix/jepa-world-models/pull/6) is still open and in draft state. It is marked mergeable and depends on PR #5. GitHub has no submitted reviews. This branch uses the PR #6 head as its base and does not change its v2 results.

The raw v1 replay and v2 records support the claims in PR #6: the retained v1 JEPA models meet the low-spread warning, the v2 exact and handcrafted controls pass the observation and label checks, and the supervised visual dynamics and JEPA fits fail the frozen v2 validation gates. The report correctly limits that finding to the tested synthetic MountainCar recipes. It does not identify a causal loss defect or a general JEPA limitation.

Before the v3 fit, a static review found two record-completeness gaps in the new v3 runner: a data-path stop needed per-arm `not_run` records, and early simulator termination needed a row for each missing scheduled start. Both are fixed and covered by tests. The protocol amendment and study-code hashes were frozen before the v3 run. No v1 or v2 records changed.

## Audit of the v2 supervised model

The v2 model uses two consecutive 48 × 128 RGB frames, ordered previous then current. It converts pixels from uint8 to float32 in [0, 1] and concatenates the frames into six channels. Its 32-dimensional visual encoder feeds an action head with `Linear(33, 64)`, GELU, and `Linear(64, 2)`. It predicts absolute next position and velocity with raw equal-weight MSE. Training uses AdamW at 0.002, weight decay 0.0001, batch size 256, and 35 epochs.

The raw v2 fit records show that all three supervised fits completed but failed their frozen validation gate. Position MAE was 0.0261–0.0803; velocity MAE was 0.00424–0.00704. No supervised visual prediction clipped. Position action order varied from 0.467 to 1.0 and velocity action order from 0.50 to 0.95. Some seeds preserved action order, but action-contrast magnitude and state error still failed.

The v2 state probes show position R² of 0.914–0.968 and velocity R² near zero. Embedding effective rank was 1.01–1.16. These are confirmed model diagnostics. They do not show whether the encoder, action predictor, state readout, loss scale, or their interaction caused the failure. V2 gradient evidence is aggregate and does not provide the per-coordinate v3 gradient measurements.

The v2 next-position target variance was about 140 times next-velocity variance. This makes target-scale imbalance a plausible explanation under raw equal-weight MSE. V2 did not compare raw and standardized targets with all other settings held fixed, so the variance ratio is not a causal result.

## Frozen v3 study

The protocol fixes data seeds 9701, 9702, and 9703, paired model seeds 9801, 9802, and 9803, and 20 validation action contexts per data seed. Each group contains 4,800 training rows and 1,200 validation rows. The action-shuffling control, learned state estimator, raw-target visual model, target-standardized ablation, and residual model use the same training and validation data.

The hierarchy keeps five controls distinct:

1. Handcrafted visual state plus exact simulator dynamics.
2. Ridge regression from the same handcrafted position, velocity, and action features.
3. A learned visual estimator for current position and velocity.
4. The existing action-conditioned visual model.
5. A visual residual-dynamics alternative.

The protocol also includes a fixed action-shuffling negative control. Thresholds, split, model/data seeds, action contexts, training budget, and stop rules were locked before training. See the [protocol](protocols/visual-jepa-supervised-positive-control-v3.md) for all values.

## Raw validation results

### Data and observation controls

| Data seed | Exact label max error | Handcrafted position MAE | Handcrafted velocity MAE | Position / velocity action order | Outside bounds | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 9701 | 5.96e-8 | 0.0037484 | 0.0000564 | 1.00 / 1.00 | 0% | Pass |
| 9702 | 5.96e-8 | 0.0037487 | 0.0000569 | 1.00 / 1.00 | 0% | Pass |
| 9703 | 5.96e-8 | 0.0037520 | 0.0000552 | 1.00 / 1.00 | 0% | Pass |

The observations are 48 × 128 RGB, previous frame then current frame, scaled by 1/255, and resized with Pillow bilinear sampling. The handcrafted velocity uses the difference between the two frame-based position estimates. These controls confirm that this observation format and label path contain the state and action information for this task.

### Completed one-step controls

The table reports raw MAE over all 1,200 validation transitions in each group.

| Data seed | Handcrafted state + exact position MAE | Exact velocity MAE | Handcrafted ridge position MAE | Ridge velocity MAE |
|---:|---:|---:|---:|---:|
| 9701 | 0.0037214 | 0.0000603 | 0.0385690 | 0.0032098 |
| 9702 | 0.0037219 | 0.0000607 | 0.0382583 | 0.0033044 |
| 9703 | 0.0037249 | 0.0000597 | 0.0345096 | 0.0030625 |

The exact control stays close to the handcrafted state-estimation error. The fixed linear ridge model is worse on both coordinates. On the 20 shared action contexts, ridge position action order was 1.00, 0.00, and 1.00 by data seed; velocity order was 1.00 for all seeds. Its median normalized action-contrast errors were 0.1949–0.5352 for position and 0.2149–0.2657 for velocity. Neither control clipped. The linear ridge result shows that these features plus this fixed linear map are not enough for accurate nonlinear transition prediction. It does not show what a learned visual model can achieve.

### Target scales

| Data seed | Train next-position variance | Train next-velocity variance | Ratio |
|---:|---:|---:|---:|
| 9701 | 0.0120852 | 0.00008749 | 138.13 |
| 9702 | 0.0142988 | 0.00010271 | 139.22 |
| 9703 | 0.0081465 | 0.00005736 | 142.04 |

The new data reproduce a large coordinate-scale difference. The target-standardized arm did not run, so this run does not measure whether standardization improves learned visual dynamics.

### Neural run status and gates

| Record | Count | Result |
|---|---:|---|
| Exact dynamics controls | 3 | Complete |
| Handcrafted-feature ridge controls | 3 | Complete |
| Neural fits completed | 0/15 | None |
| First learned fit | 1 | Failed during CUDA backward before `optimizer.step()` |
| Later neural fits | 14 | `not_run`, with the same stop reason |
| Data-path gate | 3/3 groups | Pass |
| Candidate model gate | — | Not evaluated; records incomplete |
| Independent test / rollouts / planning | — | Closed |

The run summary records 47.02 seconds total and 0.00 training seconds. It records the RTX 5070, PyTorch 2.10.0+cu128, and CUDA 12.8. The CUDA peak allocation was not recorded because the first fit stopped during backward. We therefore do not claim that the 2 GiB memory limit was verified for the incomplete neural run.

## What is confirmed and what remains open

| Finding or explanation | Evidence | Status |
|---|---|---|
| The v3 labels match exact simulator dynamics. | Maximum error 5.96e-8 in all three groups. | Confirmed. |
| Two rendered frames contain enough position and velocity signal for a fixed estimator. | Handcrafted MAE and action-fidelity gates pass in all groups. | Confirmed for this renderer and task. |
| A fixed linear map from handcrafted state and action is accurate enough for next-state prediction. | Ridge MAE and position action-order results fail on the recorded validation data. | Not supported for this fixed ridge configuration. |
| Position/velocity target scale affects the learned model. | Next-state variance ratio is 138–142. | Plausible; the standardized model did not train. |
| A learned encoder fails to retain velocity. | V2 velocity probes are near zero, but the v3 learned state estimator did not complete. | Plausible from v2; not isolated in v3. |
| The v2 action head ignores actions. | V2 action ordering is mixed. The v3 learned action models did not train. | Not confirmed. |
| Clipping caused the supervised v2 failure. | V2 supervised predictions and the v3 completed controls have zero clipping. | Not supported by available records. |
| Gradient imbalance caused the v2 failure. | V2 does not have coordinate-specific gradient evidence. The v3 backward call stopped before training diagnostics. | Unknown. |
| Autoregressive errors or planning OOD caused failure. | No v3 model passed validation; no held-out rollouts were run. | Not measured. |
| The CUDA deterministic guard prevented v3 training. | The first backward call reports no deterministic CUDA implementation for adaptive average-pool backward. | Confirmed execution blocker for this run only. |

The v3 run does not explain the cause of the v2 supervised visual model's failure. It validates the data path and gives one informative result about the handcrafted ridge baseline. It provides no learned visual embedding, gradient, action-sensitivity, clipping, or learning-curve evidence.

## Evidence and reproducibility

The raw run records preserve all planned arms. Every not-run arm has a stop reason. The evidence validator reports 21 model-run rows, 360 action-triplet rows, 17 verified evidence files, `status=incomplete`, and `heldout_open=false`.

The v3 figure package contains two deterministic SVG/PNG figures. Figure A shows the completed exact and ridge action responses on the 20 shared contexts for each seed. Figure B shows per-coordinate one-step MAE, target variance ratios, and the 0/15 neural-fit status. Both figures state their sample sizes and limits.

### Repository checks

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -q` | Pass: 91 tests. |
| `ruff check src scripts tests` | Pass. |
| `uv lock --check --offline` | Pass. |
| `uv build --offline` | Pass: source distribution and wheel built to the E: staging drive. |
| v3 evidence validation | Pass: 17 evidence files; 21 run rows and 360 action rows; held-out data closed. |
| v3 figure regeneration check | Pass: both retained figures regenerated with matching hashes. |
| `pytest` runner | Not available in the locked environment; the repository suite uses `unittest`. |
| CPU neural-fit smoke | Not run. The frozen v3 fit stopped before an optimizer update, and we did not retry under a changed device or deterministic setting. |

These checks validate code packaging and retained records. They do not turn the incomplete neural run into a completed scientific result. The CUDA memory cap was not verified, and no model learning curves or neural action-fidelity measures were produced.

Regenerate the retained figures with:

```bash
uv run --locked --offline --extra dev python scripts/plot_visual_jepa_supervised_positive_control_v3_observed.py
uv run --locked --offline --extra dev python scripts/plot_visual_jepa_supervised_positive_control_v3_observed.py --verify-determinism
```

The CUDA run command and evidence checks are recorded in the run summary and this report. Do not rerun into the same evidence directory. A later attempt requires a new, reviewed protocol version or amendment, a new output directory, and a new lock before training. It must state how deterministic execution will be supported before any model fit begins.

The PR #6 v2 evidence remains unchanged. This report is a local synthetic MountainCar study, not a paper reproduction and not a claim about JEPA in general.
