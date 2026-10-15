# RoboJEPA Candidate Preflight

**Reviewed**: 2026-10-15

**Status**: Research candidate; no run started

## Scope

RoboJEPA is a new candidate for a future paper-reproduction lesson. The paper reports that held-out latent rollout error follows a second-order scaling law with training compute. It also reports that planning success improves with compute and that a pretrained model can plan toward a goal image.

The first-party repository publishes code and checkpoints from 22M to 8B parameters. It describes two evaluation paths:

- DROID holdout: fixed clips and per-step latent L1 for steps 1–9. A scaling-law check needs results across model sizes and checkpoints, not one checkpoint.
- RoboCasa planning: four named tasks. The released scripts run one episode per seed and use seeds 0–4. A single task checks only that task's result.

Do not use either path to claim that one checkpoint validates the full scaling law.

## Source and asset gates

| Item | First-party record | Current status |
| --- | --- | --- |
| Paper | [RoboJEPA, arXiv:2610.10515](https://arxiv.org/abs/2610.10515), v1 submitted 2026-10-07 | Reviewed; the current report is based on v1 |
| Code | [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa) | Repository reviewed; its current commit is not pinned in this project |
| Model weights | Public 22M–8B predictor checkpoints and per-size configs are linked in the repository README. An HTTP HEAD check on 2026-10-15 reports 102,192,971 bytes for the [22M 150k predictor](https://dl.fbaipublicfiles.com/robojepa/wm/wm-22m/cooldown_22m_150k.pth.tar). | No weight downloaded; no local hash recorded |
| Encoder | The V-JEPA 2.1 ViT-G/384 encoder is required for every world model. An HTTP HEAD check on 2026-10-15 reports 30,238,058,912 bytes for the [official encoder checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt), about 28.2 GiB. | No encoder downloaded; no local hash recorded |
| Repository license | CC BY-NC-SA 4.0; third-party components retain their own terms | Reviewed; check every asset and upstream dependency before use |
| DROID data | The [official holdout evaluator](https://github.com/facebookresearch/robo_jepa/blob/main/scripts/evals/README.md) uses converted episodes and fixed clips | No data downloaded; identify the exact holdout files and their terms before transfer |
| RoboCasa | The [official planning guide](https://github.com/facebookresearch/robo_jepa/blob/main/docs/planning.md) lists about 23 GB of kitchen assets | No simulator assets downloaded |

## Local resource gate

The official environment pins Python 3.11 and PyTorch 2.7.1 with CUDA 12.6. The host has a 12 GB RTX 5070 and roughly 29 GiB available system memory. E:-hosted CUDA runtimes are available under Python 3.10/PyTorch 2.7.0+cu128 and Python 3.13/PyTorch 2.10.0+cu128, but they do not match the official stack. The 28.2 GiB encoder is larger than GPU memory and close to system memory. Loading or offload is unverified, so the host is not ready for official GPU evaluation. A smaller predictor does not remove the encoder requirement.

The official planning guide estimates about 20 minutes per Reach episode and about 3 hours per ObjectPush episode for the default eager 50M planner on an H100. It says the shipped scripts differ from the paper for some tasks: ObstacleReach uses five rather than 18 steps; ObjectReach does not search the gripper; and Reach and ObjectReach use a pretraining kitchen and different start-goal distances. Check the selected task's full settings before comparing scores.

## Stop conditions

Before creating a run feature or transferring assets:

1. Pin the repository commit, model config, predictor checkpoint, and V-JEPA encoder. Record their sizes and SHA-256 hashes.
2. Review terms for the checkpoint, encoder, DROID data, RoboCasa assets, and installed third-party components.
3. Select one exact paper claim, task, metric, split, seeds, planner settings, baseline, and acceptable deviations.
4. Measure whether the chosen evaluation fits an available CUDA environment, storage budget, and time budget. The current host preflight fails the CUDA runtime gate.

No checkpoint, dataset, simulator asset, or upstream checkout was transferred. No training or evaluation was run. See the [roadmap](../ROADMAP.md) and [reproduction matrix](../REPRODUCTION_MATRIX.md) for this candidate's place in future work.
