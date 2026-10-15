# Planning Limits: Local Teaching Analogue

**Status**: Locally measured illustrative analogue; not a reproduction of the paper.

## Paper-reported results

Version 1 of [The Planning Limits of Latent World Models](https://arxiv.org/html/2609.39235v1) reports 16 Meta-World tasks and 20 paired episodes per task for its closed-loop study. With V-JEPA 2.1, the paper reports 23% success for pure imagination at a 5-step rollout, 30% for 5-step feedback MPC, 52% for 30-step feedback MPC, and 76% for 5-step MPC with nearby expert subgoals. The expert subgoals come from a rollout of the same episode. These values describe the paper's setup only.

## Local run

Notebook: [10_planning_limits.ipynb](../../notebooks/10_planning_limits.ipynb)

| Item | Recorded setting |
| --- | --- |
| Environment | Gymnasium `MountainCarContinuous-v0`, Gymnasium 0.29.1 (within the project dependency range) |
| Start states | Gymnasium default reset range; 20 paired seeds, 1000–1019 |
| Goals | Positions 0.0, 0.25, 0.45 |
| Success | Position at or beyond the goal with non-negative velocity |
| Episode limit | 300 environment steps |
| Model | Exact state transition, checked against Gymnasium; maximum absolute error 0.0 on four fixed cases |
| Planner | CEM, 64 candidates, 8 elites, 2 iterations; execute one action and replan |
| Rollout horizons | 5 and 20 environment steps |
| Score | Closest predicted state: `(position - target)^2 + 0.05 * velocity^2`, plus `0.001 * mean(action^2)` |
| Pure imagination | Replans from its own predicted state and does not read later environment observations |
| Feedback MPC | Replans from the latest environment observation after each action |
| Subgoal MPC | Targets the state five steps ahead on a same-seed reference path from the momentum rule |
| Momentum rule | Apply +1 force when velocity is non-negative; otherwise apply -1 |
| Compute | CPU only; WSL2 x86_64; 24 logical CPUs; no GPU used |
| Python / NumPy | Python 3.13.14 / NumPy 2.5.4 |
| Missing/error runs | 0 of 420 planned episodes |
| Evaluation time | 71.0 seconds for the evaluation loop; notebook startup is not included |

The reference path is a scripted teaching aid. It is not an expert demonstration and uses information that pure imagination and feedback MPC do not receive. The local result therefore illustrates the value of a nearby target; it is not a fair claim that the subgoal planner is better without that guidance. Exact dynamics also remove the model errors that make feedback important in real systems.

## Local measurements

Each cell below gives success count out of 20. Step counts are means across successful episodes only. The momentum rule has no rollout horizon and was run once per goal.

| Goal | Pure imagination H5 / H20 | Feedback MPC H5 / H20 | Reference subgoal MPC H5 / H20 | Momentum baseline |
| --- | --- | --- | --- | --- |
| 0.00 | 0 / 0 | 0 / 0 | 11 (104.4 steps) / 11 (119.5 steps) | 20 (71.5 steps) |
| 0.25 | 0 / 0 | 0 / 0 | 20 (109.2 steps) / 20 (127.0 steps) | 20 (104.2 steps) |
| 0.45 | 0 / 0 | 0 / 0 | 20 (113.0 steps) / 20 (135.8 steps) | 20 (107.2 steps) |

The Wilson 95% intervals are 0.0%–16.1% for 0/20, 34.2%–74.2% for 11/20, and 83.9%–100.0% for 20/20. Every planner method made 300 calls on its failed 0/20 conditions. For subgoal MPC, mean planning calls equal mean episode steps because it replans after each action. The momentum baseline makes zero planning calls.

Raw outcomes use seed order 1000 through 1019. `F` means a completed episode without success; `S` means success; `M` means a missing or errored run.

- Pure imagination and feedback MPC: `FFFFFFFFFFFFFFFFFFFF` for every goal and horizon.
- Reference subgoal MPC at goal 0.00, for both horizons: `SSSFFFSFFFFSSSSSFSSF`.
- Reference subgoal MPC at goals 0.25 and 0.45, for both horizons: `SSSSSSSSSSSSSSSSSSSS`.
- Momentum baseline at all goals: `SSSSSSSSSSSSSSSSSSSS`.
- Missing or errored runs: none.

## Interpretation and limits

Under this fixed CEM budget and closest-distance score, pure imagination and feedback MPC did not reach any final goal in 300 steps. The momentum baseline reached every goal. The scripted reference subgoals let CEM reach the 0.25 and 0.45 goals on these seeds, but did not improve the 0.00 result to 20/20. A 20-step rollout did not improve success over a 5-step rollout in this sweep and took more steps on successful subgoal runs.

These measurements depend on the environment, score, random seeds, and CEM budget. They do not match the paper's Meta-World success rates. They do not measure visual prediction, latent representations, robot data, or physical robot control. The 20 episode intervals do not capture training-run variability because this lesson trains no model.
