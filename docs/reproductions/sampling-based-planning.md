# Sampling-Based Planning Tutorial: Local Run

**Claim status**: Illustrative state-space analogy. This is not a paper reproduction.

**Reviewed**: 2026-10-15

## Run conditions

| Field | Value |
| --- | --- |
| Task | Hand-authored planar point mass, one rectangular obstacle, fixed goal beyond the obstacle |
| Dynamics | Known deterministic position/velocity transition with nine bounded acceleration actions |
| Task settings | World bound ±1.35; obstacle `x=[-0.22, 0.22]`, `y=[-0.46, 0.46]`; starts use `x=-1+U(-0.035,0.035)`, `y=U(-0.06,0.06)`, and velocity `U(-0.01,0.01)` per axis; goal at `(1, 0)` with position radius 0.16 and speed at most 0.11; speed cap 0.15; action acceleration levels `{-0.035, 0, 0.035}` on each axis |
| Methods | Simplified RRT, projected-grid KPIECE, categorical CEM shooting |
| Paired evaluation | 100 shared starts; five groups of 20; group seeds 2101–2105; all three methods use each start |
| Training seeds | Not applicable; no model is trained |
| Replanning | Execute at most two actions, discard planner state, and replan from the current state |
| Planning budget | At most 256 one-step transition calls per decision; horizon 16; execute two actions before replanning; RRT goal bias 0.15; KPIECE grid width 0.25; CEM uses 8 sequences × 2 iterations |
| Episode limit | 80 valid executed environment steps |
| Primary metric | Closed-loop success: enter the goal radius while below the speed limit |
| Other metrics | Executed steps, planner transition calls, per-episode runtime, five group rates, missing/error counts |
| Runtime | Fresh CPU kernel in the existing E:-hosted uv environment; Python 3.13.14, NumPy 2.5.4, Matplotlib 3.11.2; no GPU |
| External assets | None downloaded; no new dependency added |

## Locally measured outcomes

The notebook completed 100 planned episodes per method, 300 total. All methods used the same starts and the same maximum transition-call budget per decision. The tree planners often stopped before the cap after reaching a goal state; CEM often used fewer calls when a candidate hit an invalid transition. Actual calls are shown below as per-episode means.

| Method | Success | Group SD | Mean steps | Mean transition calls / episode | Mean planner time / episode | Errors | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RRT | 89/100 (89%) | 12.4 percentage points | 26.45 | 2,792.88 | 0.115 s | 0 | 0 |
| KPIECE | 26/100 (26%) | 9.6 percentage points | 18.26 | 2,758.66 | 0.065 s | 0 | 0 |
| CEM | 43/100 (43%) | 21.4 percentage points | 65.40 | 6,129.12 | 0.137 s | 0 | 0 |

Group SD is the sample standard deviation of the five group success rates, not a confidence interval or training-seed variation.

| Group seed | Shared starts | RRT | KPIECE | CEM |
| ---: | ---: | ---: | ---: | ---: |
| 2101 | 20 | 70% | 20% | 30% |
| 2102 | 20 | 100% | 25% | 55% |
| 2103 | 20 | 100% | 30% | 45% |
| 2104 | 20 | 85% | 40% | 70% |
| 2105 | 20 | 90% | 15% | 15% |

All incomplete outcomes remain in the denominator. RRT had 11 invalid planned-action failures, KPIECE had 74, and CEM had 9 invalid planned-action failures plus 48 episode-limit failures. There were no caught execution errors or missing episode records.

## Paper-reported results kept separate

The paper reports that its methods use a frozen DINO-WM predictor with image features and proprioception, nearest-neighbor RRT or projection-guided KPIECE, an action prior, batched expansion, and optional CEM polishing. Its point_maze comparison reports RRT at 49/50 successes and 0.61 s median time, versus CEM at 50/50 and 16.07 s. On Push-T, it reports RRT at 40/50 and 0.72 s, versus CEM at 46/50 and 13.33 s. These are source-reported values. The paper calculates time, cost, and transition-node summaries over solved trials. See [arXiv v1](https://arxiv.org/abs/2610.12910) and its [full text](https://arxiv.org/html/2610.12910v1).

## Limits and next gate

- The state, obstacles, transition model, actions, and goal are hand-authored and fully known to the planners.
- The notebook has no image encoder, JEPA representation, learned transition function, or model error.
- The implementations are small CPU teaching versions. They do not use the paper's action prior, batched GPU propagation, task sets, baseline configurations, or robot.
- The equal local transition-call cap is a tutorial choice. It does not match the paper's budget or hardware.
- The outcomes depend on this task, grid width, action set, horizon, goal bias, CEM population, and seeds. The five group rates do not establish a general planner ranking.
- The arXiv submission is listed as under review. The author lab page links a PDF and video, but this review found no official code or checkpoint link. Keep paper reproduction **not run** until the exact author artifacts, task data, evaluation settings, and license terms are available and reviewed.

The fresh-kernel evidence is recorded for the notebook cells and the generated in-memory episode table. Raw episode rows are not committed.
