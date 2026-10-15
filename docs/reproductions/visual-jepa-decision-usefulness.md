# Learned Visual JEPA Decision-Usefulness Lesson

**Evidence status**: exploratory validation only; the planned comparison is not run.
**Reviewed**: 2026-10-15

## Question

Does an action-conditioned visual predictor help choose actions in MountainCar, and how does its prediction quality relate to action ranking and closed-loop results?

Notebook 22 defines a local comparison between oracle dynamics, a privileged state model, a pixel-reconstructing latent model, an action-conditioned visual JEPA, direct control, and JEPA planning. The small study is not a reproduction of V-JEPA 2, LeJEPA, or another paper.

## Validation-only GPU run

The first full-size model run used the fixed training and validation split. It did not use intervention or evaluation starts.

| Item | Value |
| --- | --- |
| Environment | Synthetic MountainCarContinuous frames |
| Training data seed / model seed | 1000 / 2021 |
| Split | 16 training episodes / 4 validation episodes |
| Rows | 4,800 training / 1,200 validation |
| Model and objective | 32-dimensional visual encoder; predictive MSE plus Epps–Pulley SIGReg, weight 0.1 and 128 projections |
| Training | 35 epochs, batch size 256, float32 |
| Device | NVIDIA RTX 5070, CUDA 12.8; PyTorch 2.10.0+cu128 |
| Training time / peak allocated GPU memory | 13.02 s / 509.0 MiB |
| Validation prediction MSE | 0.001351 |
| Validation embedding spread | Mean per-dimension standard deviation 0.0000739; all 32 dimensions were below the fixed `1e-3` threshold |
| Collapse screen | Failed; the representation is collapsed under the predeclared rule |

The prediction loss is low, but the representation fails the collapse screen. This run does not support a claim that the learned JEPA predicts useful features. A low prediction loss alone is not enough. The full model record is in [visual-jepa-model-runs.csv](../evidence/visual-jepa-model-runs.csv).

A second full-size validation-only pilot used a temporary copy of Notebook 22's training path with SIGReg replaced by a local VICReg-style variance and covariance penalty. The published notebook was not changed. The pilot used the same data split, one training seed, and 35 epochs on the RTX 5070. It also collapsed: all 32 dimensions were below the threshold, with mean standard deviation `0.0001205`; validation prediction MSE was `0.05560`. Training took 11.13 seconds and used 389.6 MiB peak GPU memory. This is an exploratory adaptation, not a VICReg or LeJEPA reproduction. Both full-size pilot records are in [visual-jepa-validation-pilots.csv](../evidence/visual-jepa-validation-pilots.csv).

The validation-only probes also tried SIGReg weights 1 and 10, a non-affine output normalization, a variance-floor term with weight 1, learning rates `5e-4` and `2e-4`, and adaptive max pooling. These small runs also collapsed. Their records are in [visual-jepa-validation-pilots.csv](../evidence/visual-jepa-validation-pilots.csv). They use the same data seed and model seed for diagnosis. They are not independent training runs and are not a hyperparameter comparison.

## Pre-gate exploratory five-seed CUDA run

This run used the planned seed groups before I noticed the project ledger requires a passing pilot before final training. Preserve it as exploratory evidence and exclude all 15 rows from any final comparison. The five data/model pairs each used 20 generated episodes, with 16 training episodes (4,800 rows) and 4 validation episodes (1,200 rows). All 15 fits completed on the RTX 5070 with float32. The run did not use intervention or episode cases.

| Model seed / data seed | JEPA validation prediction MSE | Mean validation embedding standard deviation | Collapse screen |
| --- | ---: | ---: | --- |
| 2121 / 1100 | 0.003267 | 0.0000109 | Failed; 32/32 dimensions below `1e-3` |
| 2127 / 1101 | 0.002780 | 0.0000116 | Failed; 32/32 dimensions below `1e-3` |
| 2139 / 1102 | 0.001170 | 0.0000813 | Failed; 32/32 dimensions below `1e-3` |
| 2153 / 1103 | 0.000965 | 0.0000550 | Failed; 32/32 dimensions below `1e-3` |
| 2163 / 1104 | 0.001448 | 0.0000580 | Failed; 32/32 dimensions below `1e-3` |

| Model | Sum of per-model training time across five seeds | Peak allocated GPU memory |
| --- | ---: | ---: |
| Privileged state model | 24.65 s | 224.7 MiB |
| Image reconstruction model | 58.23 s | 549.8 MiB |
| Action-conditioned JEPA | 56.72 s | 526.2 MiB |

Training-time sums exclude data generation. Validation losses use different objectives and units across model families. The five JEPA rows show a repeated local training failure, not a decision-usefulness result. These seeds are now excluded from the final protocol because their results have been observed before the pilot gate passed. Their records are in [visual-jepa-training-diagnostics.csv](../evidence/visual-jepa-training-diagnostics.csv); the earlier development pilot remains in [visual-jepa-model-runs.csv](../evidence/visual-jepa-model-runs.csv).

The pre-gate run used no intervention or episode cases. After the gate was implemented, a matching CUDA pilot record was added with training protocol hash `e4df0f738d2ea537c92ffc008ff1c8d49e4238a3798be8812390937359113ba4`; it failed. A fresh full-mode setup check recognized that row and skipped all final training and evaluation. Historical probe rows without a protocol hash remain in the ledger but cannot open the gate. The final seed groups remain unopened.

## Instrumented gradient diagnosis

A validation-only replay of the development seed on CUDA found that mean validation embedding spread fell from `0.002110` at initialization to `0.000131` after one epoch. All dimensions then failed the collapse screen. At epoch 35, mean spread was `0.0000739`. On the fixed training batch, the weighted SIGReg encoder-gradient norm exceeded the prediction-loss encoder-gradient norm at each inspected point. A separate synthetic check found zero SIGReg input gradient for exactly constant zero embeddings and a smaller gradient as the embedding perturbation approached zero.

This diagnosis does not identify a confirmed defect or explain the full training collapse. The local objective differs from LeJEPA's published multi-view invariance setup and uses a different Epps–Pulley frequency range. Do not generalize this result to SIGReg or JEPA methods. The fixed gate remains closed until a matching validation-only pilot passes. The protocol-hashed CUDA pilot rerun produced the same collapse status and no final cases.

## Interpretation and limits

- No final-protocol intervention set or 20-start paired closed-loop comparison was run. A CPU smoke emitted 48 intervention rows and 12 episode rows on reserved smoke seeds; these are pipeline checks and do not meet the feature's evidence thresholds.
- The pre-gate training diagnostic is complete, but it cannot satisfy the final training stage. All five JEPA runs collapsed. The final training seeds, held-out intervention set, and paired planner evaluation remain open behind the pilot gate.
- The state model receives simulator state. It is a privileged reference, not a matched-observation baseline.
- Pixel, latent, and simulator-state scores have different units. Their values must not be pooled.
- All runs use one synthetic renderer and one simulator. The development pair is 2021/1000; smoke uses 9021/9000. The pre-gate diagnostic pairs 2121/1100, 2127/1101, 2139/1102, 2153/1103, and 2163/1104 are excluded. The reserved final pairs are 7221/5100, 7227/5101, 7239/5102, 7253/5103, and 7263/5104; they have not been run.
- GPU training works on the RTX 5070, but GPU availability does not make the failed representation valid evidence for decision quality.

## Research follow-up

LeJEPA reports a predictive objective with SIGReg, while its official implementation uses multi-view invariance and a different Epps–Pulley frequency range. This small temporal-prediction setup is only LeJEPA-inspired. VICReg uses variance and covariance terms to discourage collapsed embeddings; the local VICReg-style pilot also collapsed, so it does not validate that paper's method or claims. A 2026 VISReg preprint reports that SIGReg can have vanishing gradients near collapse and proposes separate variance and sketching terms. That is a relevant future lesson, not an explanation proven by these local runs. The paper's linked code destination returned 404 during this review; the [public checkpoint card](https://huggingface.co/BooBooWu/visreg) lists CC BY-NC 4.0. No code or weights were reused. See the [LeJEPA paper](https://arxiv.org/abs/2511.08544), [VICReg paper](https://arxiv.org/abs/2105.04906), and [VISReg paper](https://arxiv.org/abs/2606.02572).

The SIGReg and VICReg-style adaptations failed on the fixed split. The local gradient diagnosis found a zero input gradient at exact constant zero embeddings but did not identify a confirmed code defect. The official LeJEPA implementation uses multi-view invariance and a different Epps–Pulley frequency range; this notebook is not that objective. Do not start another broad objective sweep. Any future configuration must pass a protocol-matched, validation-only pilot before the reserved final model, intervention, or episode seeds are used. Preserve failures and report training-seed variation separately from evaluation-start variation.

The `1e-3` collapse threshold was set before the pilot. The decision to stop before full intervention and episode evaluation follows the pilot result; this report and the feature spec now make that stop rule explicit for future runs. No held-out intervention or episode results were used to choose the objective.

## Local validation status

- **Fresh CPU smoke after the record-contract update**: passed on 2026-10-15 in a fresh Jupyter kernel. It executed all notebook code cells with training-data/model seeds `9000/9021`, intervention seed `2900`, and episode seeds `3900/3901`. It wrote 3 complete model records, 48 complete intervention records, and 12 complete episode records to `E:\codex-research\jepa-world-models-tutorial\smoke-contract`. The intervention rows record prediction space, scoring space, and device. These are pipeline checks only.
- **Seed isolation correction**: the first smoke used intervention seed `2000` and episode seeds `3000/3001`, which overlapped an early draft of the final ranges. Only row counts and completion statuses were checked; outcome values were not inspected or used for objective selection. Those outputs are excluded. The pre-gate training pairs are also excluded; replacement final ranges are disjoint from development, smoke, and pre-gate runs.
- **CUDA validation pilot**: the protocol-hashed pilot mode ran on the RTX 5070 with PyTorch 2.10.0+cu128/CUDA 12.8. It used the fixed 16/4 split for 35 epochs; all 32 embedding dimensions failed the collapse screen. It appended a validation-only record and collected no intervention or evaluation cases. The full-mode gate check then skipped final training and evaluation. The earlier pre-gate five-seed run is excluded. The VICReg-style probe used a temporary notebook copy.
- **Focused unit checks**: 12 passed, including validation of complete, incomplete, empty, malformed, duplicate, and unpaired intervention records.
- **Notebook and evidence checks**: the notebook JSON parsed and all 9 code cells compiled. The validator accepted all 48 smoke intervention rows, including the method-specific space and device labels. The saved evidence contains 1 development model row, 15 excluded pre-gate training-diagnostic rows, and 11 validation-pilot rows. All 15 diagnostic rows have `final_comparison_eligible=false`; the latest pilot row stores the current protocol hash and failed the collapse screen.
- **Lint and packaging**: Ruff passed on the changed source and test files. `uv build` passed; the wheel and source archive were written to `E:\codex-research\jepa-world-models-tutorial\package-check-final`.
- **Portfolio checks**: the manifest parsed and its local paths resolved. All local Markdown links in the case study and portfolio README resolved. Both figures were regenerated to E: and matched the tracked images pixel for pixel. The full locked test suite passed (17 tests); a direct system-Python test invocation first failed because that interpreter did not include Matplotlib, then passed under the project `uv` environment. The final package build was written to `E:\codex-research\jepa-world-models-tutorial\package-check-portfolio-update`.
- **Spec Kit cross-artifact check**: all 11 functional requirements and 6 success criteria are represented in the 21 tasks. No constitution conflict was found. Convergence found one untracked evidence step: the final 20-start, three-action intervention set. It is now tracked as T021. T006, T007, T012, T013, and T021 remain open behind the failed pilot gate.
- **Not run**: final training on the reserved replacement seeds, the held-out intervention set, the 20-start paired comparison, and CUDA planner evaluation. The protocol-hashed pilot failed the collapse screen, so full mode skips final model training and evaluation.

## Reproduction

Use the project setup in the [README](../../README.md), then open [Notebook 22](../../notebooks/22_visual_jepa_decision_usefulness.ipynb). The locked CPU environment remains supported. On a compatible CUDA host, the notebook probes CUDA and uses it when an operation succeeds; otherwise it falls back to CPU. The tested project environment and pilot outputs were stored under E: to avoid filling the system drive. No external data, checkpoint, or robot hardware was used.
