# Semigroup-JEPA Gravity Generalization Preflight

**Reviewed**: 2026-10-15
**Status**: Research candidate; no data, checkpoint, or run started

## Selected paper result

Target Table 8's 44-step prediction on the 2D square dataset. Compare SG-JEPA (GRU), SG-JEPA (SSM), Original LeWM, and DINO-WM on position, velocity, and cumulative-rotation error. The paper reports these values:

| Method | Position L2 (m) | Velocity L2 (m/s) | Cumulative rotation MAE (turns) |
| --- | ---: | ---: | ---: |
| SG-JEPA (GRU) | 1.034 | 1.915 | 0.236 |
| SG-JEPA (SSM) | 1.148 | 2.007 | 0.249 |
| Original LeWM | 2.074 | 2.631 | 0.321 |
| DINO-WM | 2.006 | 3.082 | 0.341 |

These are paper-reported values, not local results. The paper uses 128 x 128 images, 64-frame episodes, 8,000 square training episodes, and a held-out grid of 25 gravity values from -2 to 10 with 200 episodes per value. Its model configuration uses history length 20, rollout length 5, discount 0.95, batch size 64, and 20 training epochs. The official protocol reports rollout metrics at steps 5, 16, 32, and 44; a reproduction must preserve all four horizons and the paper's state probe and aggregation.

## Source and artifact gate

| Item | First-party record | Current status |
| --- | --- | --- |
| Paper | [Semigroup-JEPA, arXiv:2609.10464](https://arxiv.org/abs/2609.10464), v1 | Reviewed; results and setup above come from the submitted version. |
| Code | [sg-jepa/sg-jepa](https://github.com/sg-jepa/sg-jepa/tree/1b7794b3b3a85e9f9746636553e0d5c30d249212) at `1b7794b3b3a85e9f9746636553e0d5c30d249212` | Revision read with `git ls-remote`; no checkout created. Project code license is MIT. Review third-party terms before use. |
| Checkpoints | Official [checkpoint notes](https://github.com/sg-jepa/sg-jepa/blob/1b7794b3b3a85e9f9746636553e0d5c30d249212/CHECKPOINTS.md), [artifact manifest](https://github.com/sg-jepa/sg-jepa/blob/1b7794b3b3a85e9f9746636553e0d5c30d249212/sg_jepa/pretrained.json), and [Hugging Face dataset card](https://huggingface.co/datasets/sg-jepa/sg-jepa) | The pinned Hugging Face revision is `a0345d0101f99251029fdbe7991b73c5ff763328`. Its dataset card declares MIT and says the repository contains inference checkpoints, not trajectories. The 1.39 GB bundle includes the square SG-JEPA GRU checkpoint and probe, but not the square SSM, Original LeWM, or DINO-WM checkpoints. The full Table 8 comparison therefore needs additional verified artifacts or training. No checkpoint was downloaded. |
| Data | The [official data generator](https://github.com/sg-jepa/sg-jepa/tree/1b7794b3b3a85e9f9746636553e0d5c30d249212/data_generation) creates the MuJoCo trajectories; the checkpoint bundle does not include data. | No data generated or downloaded. Recheck simulator assets, third-party terms, split generation, and fixed seeds before a run. |
| Runtime | The official README lists Python 3.10–3.12, PyTorch 2.11.0, and CUDA 12.8 for its reported runs, and says training, evaluation, and simulation require an NVIDIA CUDA GPU. The reported runs used B200 GPUs. | The previous project `.venv` used Python 3.13.14 without PyTorch. Separate E:-hosted Python 3.10.12/PyTorch 2.7.0+cu128 and Python 3.13/PyTorch 2.10.0+cu128 environments now pass basic CUDA checks. Neither has been qualified on the Semigroup models. The host has 12 GB VRAM, compared with the paper's B200 runs, and no model memory fit was measured. |

The paper varies one supplied scalar, gravity. It does not test whether the model infers gravity from observations. The square result is a simulated prediction result; it does not establish physical-robot performance. The official checkpoint manifest identifies the available square GRU model as a seed-42 artifact. Episode counts do not replace independent training seeds.

## Stop conditions

Before any asset transfer or evaluation:

1. Review the checkpoint repository's MIT license record; verify terms for MuJoCo assets and all dependencies.
2. Pin and hash every model and source artifact. Find matching SSM, LeWM, and DINO-WM artifacts or use the official training path for the missing methods.
3. Match the square split, 25 gravity values, 200 episodes per value, state probe, four rollout horizons, and aggregation. Record any unavailable or changed seed details.
4. Qualify a CUDA runtime and storage plan for data generation and all four methods. The current host fails the runtime gate.
5. Preserve raw per-episode and per-method values outside Git. Apply the project evaluation protocol to training-run variation and deviations.

No paper result is reproduced by this preflight. See the [roadmap](../ROADMAP.md), [reproduction matrix](../REPRODUCTION_MATRIX.md), and [project Spec Kit tasks](../../specs/001-project-vision/tasks.md) for its future status.
