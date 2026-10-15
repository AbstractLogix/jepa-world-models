# Learned Visual JEPA Decision-Usefulness Lesson

**Evidence status**: Five-seed local comparison complete. This is an exploratory MountainCar study, not a paper reproduction.
**Reviewed**: 2026-10-15

## Research question

Does an action-conditioned visual predictor improve action choice, and when does accurate prediction fail to improve closed-loop decisions?

Notebook 22 compares exact simulator dynamics, a privileged state model, a pixel-reconstructing visual model, an action-conditioned visual JEPA, direct visual-momentum control, and two JEPA CEM planners. It uses synthetic rendered MountainCar frames. It does not reproduce V-JEPA 2, LeJEPA, VISReg, or another paper.

## Protocol and validation gate

The notebook used `MountainCarContinuous-v0`, goal position 0.45, a 300-step limit, two 48 × 128 RGB frames, 20 generated training episodes per data seed, and a fixed split of 16 train episodes (4,800 rows) and 4 validation episodes (1,200 rows). The full run used five paired training-data/model seeds, 35 epochs, float32, a 10-step CEM horizon, 24 candidates, four elites, and two CEM updates. It evaluated 20 held-out intervention starts under actions {-1, 0, 1}, then 20 shared episode starts per learned model seed.

Before opening those cases, the protocol-matched validation pilot had to pass the fixed `1e-3` embedding-spread screen. The VISReg-style pilot used data/model seeds 1000/2021 and the same split. It passed the screen: mean per-dimension validation standard deviation was 0.005110, with 0/32 dimensions below threshold. Validation prediction MSE was 0.0005685. Training took 13.56 seconds and used 508.54 MiB peak GPU memory on an RTX 5070 (PyTorch 2.10.0+cu128, CUDA 12.8).

The regularizer mix was 0.6, with scale, shape, and center weights set to 1, 128 projections, and epsilon `1e-4`. The pilot's mean regularizer was 33.085: scale 0.990, shape 0.096, and center 31.999. The pass means only that this one predeclared configuration cleared the fixed collapse threshold. The spread remained small, and the center term dominated. It is not evidence of a useful representation.

The objective is an independent **VISReg-style temporal-prediction adaptation** based on the paper equations. The paper uses multi-view image augmentations. This notebook uses temporal prediction. It reuses no paper code or weights and does not test the paper's claims. The matching pilot's training protocol hash was `1ef7779c4df06084166b434ffc76b7d80560b9fc62f619b10ff52329c80ae13e`.

Earlier SIGReg, VICReg-style, and pre-gate runs remain in the evidence ledger. The SIGReg pilot and temporary VICReg-style probe collapsed; the five earlier training pairs ran before the pilot gate and stay excluded from the final comparison. The gradient diagnosis found a zero SIGReg input gradient at exact zero embeddings but no confirmed implementation defect.

## Full-run evidence

All five paired data/model seed groups completed for the state model, image-reconstruction model, and JEPA. All 15 model records use CUDA. The five JEPA seeds used the VISReg-style adaptation and passed the fixed screen, with mean validation spread from 0.00279 to 0.00574 and no dimension below `1e-3`.

| Model | Training time across five fits | Peak memory | Recorded validation diagnostics |
| --- | ---: | ---: | --- |
| Privileged state model | 25.62 s total | 224.73 MiB | Total loss: `8.85e-7` to `1.38e-5` |
| Image reconstruction | 58.54 s total | 549.81 MiB | Total loss: 0.00110 to 0.00161; pixel loss: 0.00439 to 0.00645 |
| Action-conditioned JEPA | 59.95 s total | 524.67 MiB | Prediction MSE: `3.44e-5` to `8.65e-4`; spread: 0.00279 to 0.00574; 0/32 low-variance dimensions in each run |

The validation metrics use different spaces and objectives. Do not compare their values as one shared prediction score. The learned state model also receives simulator state, so it is a privileged reference rather than a matched visual baseline.

### Changed-action ranking

The notebook tested three actions at each of 20 held-out starts. The evidence contains 960 complete rows: 60 oracle rows and 300 rows for each learned method. Ranking uses each model's own score space and excludes actual-score ties.

| Method | Correct non-tied action pairs | Interpretation |
| --- | ---: | --- |
| Exact simulator dynamics | 60/60 | Privileged reference |
| Learned state model | 248/300 (82.7%) | Simulator-state score |
| Image reconstruction model | 124/255 (48.6%) | Pixel-goal score; 45 ties excluded |
| Action-conditioned JEPA | 153/295 (51.9%) | Latent-goal score; 5 ties excluded |

The state, pixel, and latent scores have different units. These percentages are descriptive and are not a direct cross-model metric. JEPA ranking varied across training seeds from 33.3% to 66.7%; the pooled ratio does not replace the five seed-level results.

### Closed-loop outcomes

All 440 episode rows completed without an error. The 20 evaluation seeds were shared across methods. Each learned model seed used the same 20 starts. Direct visual control and oracle CEM have no training seed and ran once on those 20 starts.

| Method | Successes | Per-seed detail |
| --- | ---: | --- |
| Oracle-state CEM | 0/20 | Exact dynamics did not make this CEM setup succeed |
| Learned-state CEM | 18/100 | 18/20 on seed 7239; 0/20 on each other model seed |
| Image-reconstruction CEM | 0/100 | 0/20 for all five model seeds |
| JEPA uniform-start CEM | 0/100 | 0/20 for all five model seeds |
| JEPA momentum-prior CEM | 100/100 | 20/20 for all five model seeds |
| Direct visual-momentum control | 20/20 | One controller on the shared 20 starts |

The momentum prior starts CEM from the action sign estimated from recent visual motion. The direct controller uses that same hand-written visual-momentum rule without CEM. Since direct control already solved all 20 starts, the result does not establish that JEPA improved decisions. The prior-guided JEPA planner had a mean return of 91.95 across its 100 rows; the direct controller had 89.34 across its 20 rows. This small, unbalanced descriptive difference is not an isolated or independent JEPA effect.

The notebook also paired each learned planner row with the direct controller's return on the same evaluation start. Mean return differences were -81.34 for state CEM, -93.54 for image-reconstruction CEM, -102.54 for uniform-start JEPA CEM, and +2.62 for momentum-prior JEPA CEM. Each direct-control row is reused across five learned-model seeds, so the 100 differences per trained method are paired descriptive records, not 100 independent baseline runs.

The unrestricted JEPA planner succeeded on 0/100 starts. Oracle CEM also succeeded on 0/20 starts. Together these outcomes show that the planner setup can fail even with exact dynamics, and that the hand-written prior strongly shapes the result. The run does not isolate planner search error from horizon, objective, or model error.

## Compute and storage

The full model training used the RTX 5070 with PyTorch 2.10.0+cu128, CUDA 12.8, and float32. Total recorded training time across all 15 fits was about 144.11 seconds, excluding data generation. The highest recorded model allocation was 549.81 MiB. The notebook ran all model predictions and CEM planning on CUDA; simulator stepping and rendering remain environment work outside the learned model.

Average per-episode planning time and model-step calls were:

| Method | Planning seconds | Model-step calls |
| --- | ---: | ---: |
| Oracle-state CEM | 2.507 | 144,000 |
| Learned-state CEM | 2.874 | 135,869 |
| Image-reconstruction CEM | 2.273 | 144,000 |
| JEPA uniform-start CEM | 1.894 | 144,000 |
| JEPA momentum-prior CEM | 0.746 | 55,685 |
| Direct visual-momentum control | 0 | 0 |

The learned methods use different architectures and scoring spaces. These local latency and call counts are workload measurements, not a general hardware comparison.

The executed notebook, runtime cache, and generated arrays remain on the E: drive at `E:\codex-research\jepa-world-models-tutorial\visreg-style-full-20261015`. The repository retains the small model, intervention, and episode CSV files and the two portfolio figures. No dataset or checkpoint download was needed.

## What the evidence supports

- Clearing a spread threshold is a training diagnostic, not proof that the representation ranks useful actions.
- In this run, JEPA's one-step predictive loss and non-collapsed status did not produce reliable three-action ranking or successful uniform-start CEM.
- The hand-written visual-momentum prior changed JEPA CEM from 0/100 to 100/100, but its direct-control version also reached 20/20. This shows the value of the prior in this task, not an isolated gain from JEPA.
- Exact dynamics with the chosen CEM setup also failed on 0/20 starts. Planner initialization, horizon, and objective need separate controls before assigning the failure to a learned model.
- One-step losses, action-ranking scores, success, and compute answer different questions. None alone validates a useful world model.

## Research context and limits

LeJEPA reports a predictive objective with SIGReg. Its official implementation uses multi-view invariance and a different Epps–Pulley frequency range; this notebook uses a smaller temporal-prediction objective. The local objective studies in Notebooks 02 and 03 are small CIFAR-10 studies, not the paper's benchmarks. The official V-JEPA 2.1 checkpoint path in Notebook 09 verifies loading and feature extraction on one synthetic clip; it does not test action selection.

VISReg motivates a variance and sliced-Wasserstein regularizer. The local adaptation is separate from the paper's image-view setup. Its pilot passed a fixed spread screen, but the final JEPA models still had low spread and near-chance aggregate action ranking. These measurements do not validate VISReg or establish a general JEPA result. See the [LeJEPA paper](https://arxiv.org/abs/2511.08544), [official implementation](https://github.com/galilai-group/stable-pretraining/blob/main/stable_pretraining/methods/lejepa.py), [VICReg paper](https://arxiv.org/abs/2105.04906), and [VISReg preprint](https://arxiv.org/abs/2606.02572).

The experiment uses one small synthetic renderer, one simulator, a limited action set, and five training seeds. It does not include an independently trained direct policy, a learned-policy prior, calibrated model uncertainty, held-out tasks, or an official benchmark. It is a controlled local tutorial result, not a paper reproduction.

## Prioritized next research

The next study should isolate representation value from the hand-coded prior and CEM setup. Use a fixed policy and the same training data to compare:

1. Oracle simulator dynamics.
2. Learned state-space dynamics.
3. Learned visual latent dynamics.
4. Action-conditioned JEPA dynamics.
5. Direct policy without a learned world model.
6. The same policy as a prior for model-based search, with unconstrained search as a separate condition.

Before execution, fix the task, data split, test starts, seeds, model and planner budgets, primary outcome, and compute limit. Compare one-step accuracy, action interventions, ranking, closed-loop outcomes, uncertainty, model calls, latency, and training cost. Require full raw rows, matching access conditions, a predeclared analysis, and an independent regeneration before making a research claim. Do not use this MountainCar outcome as evidence for a larger benchmark.

The official 450-episode Push-T evaluation remains pending. The full JEPA-WAM LIBERO-Plus benchmark remains out of scope for this run.

## Local validation status

- **Training and records**: 15/15 model fits, 960/960 intervention rows, and 440/440 episode rows completed. The raw rows have no recorded failures or missing outcomes.
- **CUDA**: all three learned model families trained on the RTX 5070; the full protocol hash matched the passing VISReg-style pilot.
- **Notebook pipeline**: the pilot and full run executed on the E:-hosted Python 3.10.12 environment. The earlier CPU smoke checked the short pipeline on reserved seeds; it is not part of this comparison.
- **Focused unit checks**: 23 tests passed with `uv run --locked --no-sync python -m unittest discover -s tests -v`.
- **Code and package checks**: Ruff passed on the changed source, tests, and figure generator. `uv build` produced the source distribution and wheel; build outputs were written to E:.
- **Notebook and evidence checks**: Notebook JSON is valid and all nine code cells compile. The recorded pilot hash matches the current training cells and objective modules. The final evidence contains 15 model fits, 960 paired interventions, and 440 episode records with no recorded failures or missing outcomes. The portfolio figure generator rebuilt all four figures from the evidence files.
- **Documentation and repository checks**: the portfolio manifest paths and accessibility fields passed their unit checks. Local Markdown links and `git diff --check` passed. Spec Kit analysis covered 12 functional requirements, six success criteria, three stories, and 26 tasks; it found no remaining coverage or constitution issues. Convergence found no unbuilt work and appended no task.
- **CPU scope**: the earlier CPU smoke checked the short pipeline with reserved seeds. The full five-seed comparison was not run on CPU; the measured full run used CUDA.
- **Paper-scale benchmarks**: the 450-episode Push-T run and 2,402-task JEPA-WAM benchmark were not run.

## Reproduction

Use the project setup in the [README](../../README.md), then open [Notebook 22](../../notebooks/22_visual_jepa_decision_usefulness.ipynb). The locked CPU environment remains supported. On a compatible CUDA host, the notebook probes CUDA and uses it when a synchronization check succeeds; otherwise it falls back to CPU. Set `JEPA_RUN_OUTPUT_DIR` to an E: or D: folder to keep run outputs away from the system drive. No external data, checkpoint, or robot hardware was used.
