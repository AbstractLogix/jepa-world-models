# Research: LeHami Actuated-Pendulum Velocity Reproduction

## Selected claim

The selected result is Table 1 of *LeHami: Symplectic Latent Dynamics for World Models*, arXiv:2610.15267v1, submitted 2026-10-11. On the actuated visual pendulum, the paper reports anchor (K=0) test angular-velocity MAE of **0.419 ± 0.045 rad/s** for LeHami with a 25-NN probe and **2.508 ± 0.002 rad/s** for the LeWM baseline. The table reports mean and standard deviation across three training seeds. The LeHami probe predicts velocity better; LeWM predicts angle better. This feature tests only the velocity result.

## Primary sources and artifacts

- Paper: [arXiv:2610.15267](https://arxiv.org/abs/2610.15267), especially Table 1 and Appendix D.2–D.4.
- Official implementation: [rominay/LeHami](https://github.com/rominay/LeHami), cloned at `f76f65f8b99afb6b7e3063f573645d3e2f290ede`; repository license is Apache-2.0.
- Official experiment: `lehami/experiments/actuated_pendulum.sh`; the selected arms are `a0q` and `b0s`, with training seeds `0 1 2`.
- Official model configurations: `lehami/training/configs/experiment/phase2_lehami_sg_sep_conv_w1_swingup_exp079_a0q.yaml` and `lehami/training/configs/experiment/phase0_lewm_ms_conv_w1_swingup_exp072_b0s.yaml`.
- Official split: `lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json`. It records the expected dataset SHA-256 and episode-level split.
- Shared baseline code: [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), pinned by the LeHami repository to commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.

## Matched protocol

The actuated dataset contains 900 generated episodes of 64×64 RGB frames, 80 frames per episode, three rod lengths, and three masses. Torque is held for three frames. The paper uses 630/135/135 train/validation/test episodes, split by episode and stratified across design cells. The official generator and shipped manifest use seed 42.

The official `a0q` and `b0s` configs use 35 epochs and batch size 32. The complete comparison uses three training seeds (0, 1, 2), official checkpoints, and a 25-NN distance-weighted probe fitted at K=0 on held-out test episodes. The repository helper `scripts/reproductions/lehami_k0_probe.py` calls the official feature collector and probe routine, verifies the HDF5 hash, and stores the metric, fitted probe, and test predictions. This focused comparison does not run the paper's rollout, planning, conservative-pendulum, DINO, or other probe-capacity studies.

## Interpretation limits

- The paper's exact symplecticity result rules out uniform phase-space volume contraction under its stated map. It does not bound rollout error or establish that the latent coordinates are physically correct.
- The pendulum data are generated. This result does not validate a robot or a real physical system.
- A one-epoch smoke run, frozen paper figures, and a result from fewer than all six model/seed combinations are not a reproduction of the selected three-seed table result.
- The paper reports three seeds but the table does not name them; this plan uses the official runner's seeds 0, 1, and 2 and records that basis.
- Matching the paper's qualitative direction is not enough to hide a numeric difference. Report all values and protocol details.

## Local environment setup

The first official `uv sync --locked` attempt selected Python 3.14, where the locked Box2D wheel is unavailable. Retrying with Python 3.13 reached a second blocker: `labmaze==1.0.6` builds through Bazel, pulled in by the optional `stable-worldmodel[env]` extra. The selected data, training, evaluation, and probe files do not import Gymnasium, OGBench, MuJoCo, or Box2D. A temporary runtime project therefore removed only that unused extra and generated a derived lock. All 223 package versions retained from the official lock match exactly; the derived lock SHA-256 is `ff333fef81053f63857bf6a351235bc7e24a05b29fe93dc65003c4ca4463d195`. The official LeHami source checkout and its `pyproject.toml` and `uv.lock` remain unchanged.

The derived environment uses Python 3.13.14, PyTorch 2.12.0+cu130, torchvision 0.27.0+cu130, CUDA runtime 13.0, and scikit-learn 1.8.0. `torch.cuda.is_available()` returned true for the NVIDIA GeForce RTX 5070 (12,227 MiB; driver 617.42). This environment-only adjustment is recorded as a deviation; it does not change model source or experiment settings.

## Alternatives considered

- The paper's frozen figure notebook was not selected because it reads archived values and does not independently train the models.
- The paper's swing-up planning result was not selected because it adds planner settings and closed-loop episodes beyond the minimal velocity-decoding claim.
- The conservative pendulum rollout comparison was not selected because the actuated Table 1 directly tests whether the latent carries velocity when actions change the dynamics.
