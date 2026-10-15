# Research: LeHami Seed-0 Actuated Rollout Evaluation

**Reviewed**: 2026-10-15

## Findings

- The LeHami paper's actuated-pendulum rollout figure uses a 25-neighbor probe fitted at K=1 and then applied frozen. The paper reports means and standard deviations across three architecture seeds. It describes LeHami as better on velocity early in the rollout, with performance becoming similar after about K=10; angle error becomes similar after about K=40. These are paper-reported results, not local results.
- The official evaluator's standard mode fits the K=1 probe on the training split and records probe MAE by rollout step. Its default KNN neighbor count is 25. Horizon 76 is the longest rollout for the selected 80-frame episodes after the three-frame context.
- The project already has one full seed-0 training run and a K=0 evaluation for each model. Both checkpoints and data are outside the repository. The verified actuated-pendulum data hash is `9ae8f830c1ef3e5f4a04414de9278a295817546bfd4f1b9eb4c6bfc19eafae37`.

## Decisions

### Decision: Use the pinned official standard evaluator and existing checkpoints

Run the authors' `run_eval.py` on the seed-0 `a0q` and `b0s` checkpoints with the official manifest and true recorded actions. Use CUDA, 25 neighbors, and horizon 76. Give each run a new output name under the temporary official checkout so existing outputs remain intact.

**Rationale**: This follows the paper's rollout probe setup without another training job or a new evaluator implementation.

### Decision: Keep this seed-0 rollout result separate from the three-seed claim

Report per-step values for the local seed 0 and do not aggregate it with the paper's statistics. Keep the four remaining training runs open.

**Rationale**: One run per model cannot estimate local variability or reproduce the paper's three-seed summary.

### Decision: Add measured prediction evidence but no planning or physical claim

Update Notebook 15 to show the measured rollout comparison alongside the oscillator demonstration. Keep planning and physical validation as separate evidence requirements.

**Rationale**: A frozen probe across rollouts measures prediction. It does not measure whether a planner selects useful actions or whether the model matches a real pendulum.

## Primary Sources

- Yalovetzky, Balestriero, and LeCun, [LeHami: Symplectic Latent Dynamics for World Models](https://arxiv.org/abs/2610.15267), v1. Figure 15 and its discussion describe the actuated-pendulum rollout probe; Section 3.2 distinguishes prediction from structural guarantees.
- Authors' [official LeHami repository](https://github.com/rominay/LeHami), including `lehami/experiments/src/run_eval.py` and the actuated-pendulum manifest. The repository is Apache-2.0.
- Existing checkpoint, data, and K=0 details: [project reproduction report](../../docs/reproductions/lehami-actuated-velocity.md).
