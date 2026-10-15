# JEPA World Models Curriculum Guide

This guide gives learners and instructors a clear route through the course. The [coverage map](docs/CURRICULUM_COVERAGE.md) lists all 21 lessons and their evidence limits. Read the [evaluation protocol](docs/EVALUATION_PROTOCOL.md) before comparing results.

## Recommended learning path

| Session | Lessons | Learning outcome |
| --- | --- | --- |
| 1 · Foundations | [01 · Intuition](notebooks/01_intuition.ipynb) | Explain predictive learning and identify representation collapse. |
| 2 · Representations | [02 · LeJEPA](notebooks/02_lejepa_from_scratch.ipynb), [03 · Rectified LpJEPA](notebooks/03_rectified_lpjepa.ipynb) | Compare objectives and inspect embedding quality. |
| 3 · Values and planning | [05 · Value-guided planning](notebooks/05_value_guided_planning.ipynb) | Connect learned representations to candidate action planning. |
| 4 · Gradient planning | [06 · Gradient-based planning](notebooks/06_gradient_based_planning.ipynb) | Compare random shooting with gradient-based planning. |
| 5 · Evidence and limits | [10 · Planning limits](notebooks/10_planning_limits.ipynb), [14 · Decision usefulness](notebooks/14_decision_usefulness.ipynb) | Separate prediction accuracy from closed-loop success. |

The core path does not need robot hardware or private data. Some lessons need downloads or extra packages. Check the coverage map before class or a long run.

## Optional lesson groups

| Topic | Lessons | What they add |
| --- | --- | --- |
| Earlier concept bridges | 04, 07, 08 | Vision-language ideas, latent actions, and planning ablations in small control tasks. |
| Video and object features | 09, 12, 13 | Dense video features, generated video actions, and mask-pooled features. |
| Language prediction | 11 | Predict real CIFAR-10 class-prompt embeddings. This is a supervised analogue, not a VL-JEPA reproduction. |
| Planning and reliability | 16, 17 | Action-conditioned visual prediction, uncertainty calibration, and model exploitation checks. |
| Structured and hierarchical models | 15, 18, 19 | Symplectic dynamics, multi-path inference, and hierarchical planning analogies. |
| Policy-noise planning | [20](notebooks/20_policy_noise_planning.ipynb) | Compare raw-action and policy-noise search under a deliberately imperfect toy model. |
| Sampling-based planning | [21](notebooks/21_sampling_based_planning.ipynb) | Compare RRT, KPIECE, and CEM on a hand-authored obstacle task; this is not a paper reproduction. |

These lessons use different levels of evidence. The [coverage map](docs/CURRICULUM_COVERAGE.md) states what each notebook measures and what it does not establish.

## Prepare an environment

From the repository root:

```bash
uv sync --locked
uv pip install -r requirements-torch.txt  # CUDA 12.8; omit for CPU-only use
uv run jupyter lab
```

This project uses `uv` to install and run Python packages. To keep the environment, package cache, lesson data, checkpoints, and model caches on E: or D:, follow the [external-storage quickstart](specs/024-external-asset-storage/quickstart.md) before setup. The commands select future paths; they do not move files that already exist.

The lock uses CPU PyTorch wheels by default. For CUDA 12.8, install those wheels after `uv sync --locked`. Select Notebook 11's extra before installing or replacing PyTorch:

```bash
uv sync --locked --extra vljepa
uv pip install -r requirements-torch.txt  # optional CUDA 12.8 override
```

For CPU-only setup, use the locked CPU wheels and omit the CUDA install command. `uv sync --locked` restores those CPU wheels; on a GPU host, reapply the CUDA override after each sync.

Check the notebook and coverage map before class if you need to download data or model weights. In particular, Notebook 09 downloads an approximately 1.66 GB checkpoint, Notebook 11 uses about 300 MB of first-run assets, and Notebook 13 uses about 800 MB of dataset files.

## Two-hour teaching plan

- **Session 1:** Notebook 01 and JEPA foundations.
- **Session 2:** Notebooks 02 and 03; compare objectives and diagnostics.
- **Session 3:** Notebook 05; connect features to values and planning.
- **Session 4:** Notebook 06; compare planners and discuss compute.
- **Session 5:** Notebooks 10 and 14; test planning limits and decision usefulness.
- **Advanced sessions:** Select one optional lesson group above. Use the coverage map to choose a suitable run.

Before class, run the selected notebook in a clean kernel and check its data and device needs. Keep a short backup explanation ready for optional downloads or slow training. During class, ask learners to state the claim and baseline before they run a comparison. After class, ask them to label the result and list the evidence needed for a stronger claim.

## Expected outputs

| Lessons | Output to discuss |
| --- | --- |
| 01 | A visual example of collapse and regularized representations. |
| 02–03 | Embedding diagnostics and a downstream probe for each objective. |
| 05–06 | Planner returns beside a named baseline; improvement is measured, not assumed. |
| 10 | Success by planning horizon and feedback setting, with the exact-model boundary stated. |
| 14 | Held-out action interventions and paired policy outcomes on the same starts. |
| 20 | Paired closed-loop outcomes, model calls, and off-support prediction error in an illustrative toy task. |
| 21 | Paired closed-loop outcomes, transition calls, and group variation for three toy planners. |

Use the [coverage map](docs/CURRICULUM_COVERAGE.md) for the expected output and claim boundary of every notebook.

## Prerequisites

- Basic Python and PyTorch.
- Introductory representation-learning concepts.
- Basic reinforcement-learning terms: state, action, reward, and return.

## Assessment rubric

| Area | Points | Evidence |
| --- | ---: | --- |
| Concepts | 30 | Explain JEPA prediction and its trade-offs against generative models. |
| Implementation | 30 | Run a notebook and identify its encoder, predictor, and loss. |
| Evaluation | 25 | Name the task, baseline, metric, and run conditions for a comparison. |
| Communication | 15 | Explain the result, limits, and a useful next experiment. |

## Suggested assignments

1. Change one architecture setting in Notebook 02 and record the run conditions.
2. Change the horizon or candidate count in Notebook 05 and report compute with return.
3. Add one ablation to Notebook 07 or 08. State which result the ablation could support.

## Troubleshooting

- **Out of memory in Notebooks 02 or 03:** reduce the batch size and record the change.
- **A noisy t-SNE plot:** use the linear probe and embedding statistics as well. A plot is not a performance measure.
- **No robot or MuJoCo setup:** the control lessons use Gymnasium. They teach simulated control and do not validate a physical robot.
- **Slow or unavailable optional download:** use a lesson without that checkpoint or dataset. Do not substitute its result for the planned measurement.
