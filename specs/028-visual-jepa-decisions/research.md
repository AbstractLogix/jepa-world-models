# Research: Learned Visual JEPA Decision-Usefulness Lesson

**Reviewed**: 2026-10-15

## Findings

### V-JEPA 2 and V-JEPA 2-AC

The [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) separates broad video pretraining from robot-data post-training of an action-conditioned predictor. The paper reports action-conditioned planning with less than 62 hours of unlabeled DROID robot video. The [official implementation](https://github.com/facebookresearch/vjepa2) publishes model code and checkpoints. These are the correct references for a later checkpoint evaluation; they are outside this feature's model scale and data scope.

**Decision**: Train a compact predictor from scratch on synthetic MountainCar frames. Call it a local JEPA lesson, not V-JEPA 2-AC or a reproduction.

### LeJEPA and SIGReg

The [LeJEPA paper](https://arxiv.org/abs/2511.08544) studies SIGReg with a predictive objective. The project has an Epps–Pulley SIGReg implementation in `src/jepa_tutorial/paper_losses.py` and a combined predictive loss helper in `src/jepa_tutorial/lejepa_loss.py`. This notebook uses a small temporal-prediction objective inspired by that work; it does not use the paper's full training recipe.

**Decision**: Reuse these helpers for a predictive latent objective and report the SIGReg term and embedding-spread diagnostics. The reduced network, temporal-prediction objective, quadrature range, data, and compute do not reproduce LeJEPA's experiments or prove its theory.

Validation-only GPU runs found collapse for SIGReg weights 0.1, 1, and 10. A non-affine output normalization, a variance-floor term with weight 1, learning rates `5e-4` and `2e-4`, and adaptive max pooling also failed to prevent collapse. On the fixed full-size split (16 training episodes and 4 validation episodes), the default objective produced mean per-dimension validation standard deviation `7.3868e-5`; all 32 dimensions were below the predeclared `1e-3` threshold. The run took 13.02 seconds and used 509 MiB peak GPU memory on an RTX 5070. A separate full-size validation-only probe used a temporary copy of Notebook 22's training path with predictive MSE plus VICReg-style variance and covariance terms (weights 25/25/1); it also collapsed, with validation MSE `0.05560`, mean standard deviation `0.0001205`, and all 32 dimensions below threshold. It took 11.13 seconds and used 389.6 MiB. This is a local adaptation, not a VICReg or LeJEPA reproduction. These are training failures, not comparison results. No intervention or evaluation starts were used. The records are in [visual-jepa-validation-pilots.csv](../../docs/evidence/visual-jepa-validation-pilots.csv).

The official [LeJEPA implementation](https://github.com/galilai-group/stable-pretraining/blob/main/stable_pretraining/methods/lejepa.py) combines multi-view invariance with SIGReg. Its Epps–Pulley implementation integrates over nonnegative frequencies from 0 to 3 with 17 points and symmetric weights. This notebook instead uses temporal future-prediction MSE and the project's signed `[-5, 5]` quadrature. These are explicit local adaptations, not the official LeJEPA objective or training recipe.

An instrumented CUDA replay of the fixed development split (data seed 1000, model seed 2021) found that validation embedding spread fell from `0.002110` at initialization to `0.000131` after one epoch; all 32 dimensions then failed the fixed `1e-3` screen. At epoch 35, mean spread was `0.0000739`. On a fixed batch, the weighted SIGReg encoder-gradient norm was larger than the prediction-loss encoder-gradient norm at initialization and at epochs 1, 5, and 35. A separate synthetic input check found an exact zero SIGReg input gradient for exactly constant zero embeddings; the gradient norm also decreased with the perturbation scale. These diagnostics show a local failure mode. They do not identify a confirmed implementation bug or explain the full training collapse. Do not infer that SIGReg or LeJEPA generally fails. This diagnosis predated the bounded VISReg-style pilot; the later matching pilot passed the fixed screen and the predeclared comparison then ran. No broad objective sweep was performed.

A five-seed CUDA training run was started before the project-level pilot prerequisite was noticed. All five JEPA rows collapsed. No held-out cases were collected, but these model/data seeds are now classified as pre-gate exploratory and excluded from the final comparison. The replacement final seed pairs are 7221/5100, 7227/5101, 7239/5102, 7253/5103, and 7263/5104. After the matching VISReg-style pilot passed, all five pairs were used for the predeclared local comparison and recorded in [visual-jepa-model-runs.csv](../../docs/evidence/visual-jepa-model-runs.csv). These final rows are distinct from the excluded pre-gate diagnostics in [visual-jepa-training-diagnostics.csv](../../docs/evidence/visual-jepa-training-diagnostics.csv). See the [full comparison report](../../docs/reproductions/visual-jepa-decision-usefulness.md).

The [VICReg paper](https://arxiv.org/abs/2105.04906) motivates variance and covariance terms as a way to avoid constant embeddings. This local predictive adaptation is not the paper's method or experiment, and its failure does not test the paper's claims. The local gradient diagnosis did not identify a confirmed code defect. Do not start another broad objective sweep.

### VISReg follow-up

The [VISReg preprint](https://arxiv.org/abs/2606.02572) separates scale and distribution shape. Its equations center a batch, penalize each dimension's standard deviation from one, normalize by stop-gradient standard deviation, match sorted random projections to standard-Gaussian quantiles with sliced 2-Wasserstein distance, and penalize the batch mean. The full paper objective mixes its regularizer with an invariance loss. It reports `lambda=0.6` as a starting point for small image datasets. These equations motivate one local diagnostic after the SIGReg collapse; they do not explain or validate the local result.

The paper's recipe uses multiple augmented image views. This project will adapt the regularizer to a compact temporal future-prediction loss on synthetic MountainCar frames. That change makes it a VISReg-style temporal-prediction adaptation, not a paper reproduction. The predeclared local setting is `lambda=0.6`, component weights `1`, 128 projections, a 32-dimensional embedding, and `1e-4` normalization epsilon. It receives one fixed validation-only pilot; no parameter sweep or held-out case selection is allowed.

The paper links to [haiyuwu.github.io/visreg](https://haiyuwu.github.io/visreg/), which returned 404 during this review. The author-linked [HaiyuWu/visreg repository](https://github.com/HaiyuWu/visreg) contains a README that points to [altos-labs/visreg](https://github.com/altos-labs/visreg); that destination also returned 404, and the author repository reports no license metadata. No first-party implementation or code license was available to verify. This project will implement only the published equations and will not reuse code or weights. The public checkpoint card's separate license does not grant rights to unavailable code or change this decision.

The [VIScore preprint](https://arxiv.org/abs/2608.11174) described a planning-relevance score, but arXiv marks the paper withdrawn because the submitter did not have the rights to agree to the license. Do not use its claims or metric as a research basis unless an authorized public version is available.

### V-JEPA 2.1 and RoboJEPA

The [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482) studies dense prediction, deep self-supervision, multimodal tokenizers, and scale. The [RoboJEPA paper](https://arxiv.org/abs/2610.10515) reports that latent rollout error follows a compute scaling law and that planning success improves predictably with compute. Its [official repository](https://github.com/facebookresearch/robo_jepa) releases predictors from 22M to 8B parameters. Its preflight record in this repository notes that checkpoint evaluation also requires a 28.2 GiB V-JEPA 2.1 ViT-G/384 encoder and large evaluation data.

**Decision**: Keep these findings on the future research roadmap. A five-seed local MountainCar result cannot validate dense feature quality, robot transfer, or scaling laws. Do not download those model or data assets for this lesson.

### Evaluation evidence

The [project evaluation protocol](../../docs/EVALUATION_PROTOCOL.md) distinguishes prediction, intervention, closed-loop use, decision benefit, and reliability. Notebook 16 already provides a deterministic renderer and shows why low one-step error is not sufficient. Its color-moment model is handcrafted, not JEPA.

**Decision**: Reuse its task and renderer, but use separate train, validation, intervention, and evaluation seeds. Notebook 22 now collects final intervention starts only after the locked development pilot passes. Record paired outcomes and distinguish training-seed variation from evaluation-start variation. Mark state-based methods as privileged.

## Runtime check

The host exposes an NVIDIA RTX 5070 with 12,227 MiB total. The project environment is on E: at `/mnt/e/codex-research/jepa-world-models-tutorial/.venv`, with Python 3.10.12, PyTorch 2.10.0+cu128, CUDA 12.8, and NumPy 2.2.6. The CUDA build supports `sm_120`; a convolution and the validation pilot completed on the GPU. Earlier CUDA runtimes were excluded because they had unsupported GPU architecture or did not match the project lock. Pilot outputs and caches stay on E:.

## Alternatives considered

| Choice | Decision | Reason |
| --- | --- | --- |
| Download V-JEPA 2 / RoboJEPA weights and robot data | Defer | Checkpoint, dataset, license, and compute costs exceed this local lesson. The preflight records the exact gates. |
| Keep only the color-moment transition model | Reject for this feature | It is useful as the existing analogy, but it cannot teach a learned visual representation or validate a JEPA objective. |
| Use one local seed and present the result as a comparison | Reject | The evaluation protocol requires five training seeds for a comparison. An incomplete run remains exploratory or partial. |
| Make CUDA mandatory | Reject | The core tutorial must remain usable with the locked CPU environment. CUDA is optional and selected only after an operation succeeds. |

## Limits

This design uses synthetic frames, one simple simulator, small models, one task, and one GPU host. It does not establish general JEPA decision quality, paper reproduction, robot transfer, cross-camera robustness, or a compute scaling law.
