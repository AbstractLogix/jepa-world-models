# Decision-Usefulness Lesson: Local MountainCar Evidence

**Status**: Locally measured illustrative analogue. This is not a JEPA result or a reproduction of a cited paper.

## Research basis

The decision-making position paper discusses counterfactual action fidelity, closed-loop validity, policy ranking, optimization lift, exploitability, and uncertainty calibration. The benchmark survey reports 11 explicit world-model versus direct-policy comparisons among 160 surveyed benchmarks. These works motivate the evidence types below; this local run does not reproduce their results.

Sources: [Yu et al., decision-making evaluation position paper](https://arxiv.org/abs/2606.15032); [Jena et al., benchmark survey](https://arxiv.org/abs/2609.29669).

## Fixed local protocol

| Item | Setting |
| --- | --- |
| Environment | Gymnasium MountainCarContinuous-v0, version 0.29.1; goal position set to 0.25 |
| Success | Position at or beyond 0.25 with non-negative velocity |
| Episode limit | 300 steps |
| Training data | 6,000 simulator transitions from random actions; seeds 1000–1019 |
| Intervention cases | 20 states from five held-out random-action episodes, seeds 2000–2004; actions -1, 0, and 1 from each state |
| Evaluation | 20 paired starts, seeds 3000–3019, identical across both methods |
| Learned model | Degree-3 polynomial features and deterministic Ridge regression, alpha 1e-6 |
| Planner | CEM over the learned model; horizon 10, 64 candidates, 8 elites, 2 iterations; replan after each action |
| Direct policy | Apply +1 force when velocity is non-negative; otherwise apply -1, matching the Notebook 10 momentum rule |
| Compute limit | Same 60-second elapsed-time cap for each 20-episode policy evaluation |

The training, intervention, and evaluation seed groups are disjoint. The local simulator transition helper matched Gymnasium within 1.9e-9 maximum absolute error over four fixed states.

## Intervention results

The notebook applies all three actions from each held-out state in both the simulator and fitted model. It preserves all 60 predicted and observed next states in its output table.

| Measure | Result |
| --- | ---: |
| Intervention outcomes | 60 |
| Held-out states | 20 |
| Mean absolute position error | 2.2852e-7 |
| Mean absolute velocity error | 2.7162e-7 |
| Pairwise utility-ranking agreement | 60/60 (100%) |
| Non-tied action pairs | 60 |
| Excluded tied pairs | 0 |

One-step action ranking was accurate on these sampled states. This measurement does not establish that the planner will select useful long-horizon actions.

## Closed-loop results

| Method | Success | Wilson 95% interval | Mean return | Mean steps | Model calls | Elapsed time | Within cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Learned-model MPC | 0/20 (0%) | 0.0%–16.1% | -20.74 | 300.0 | 7,680,000 | 12.747 s | Yes |
| Direct momentum policy | 20/20 (100%) | 83.9%–100.0% | 89.59 | 104.2 | 0 | 0.007 s | Yes |

There were 20 paired evaluation starts and no missing episodes. The mean paired success difference (MPC minus direct policy) was -1.00, with a paired bootstrap 95% interval of [-1.00, -1.00]. The mean paired return difference was -110.324, with a paired bootstrap 95% interval of [-110.406, -110.240]. The interval uses 10,000 resamples of the 20 paired starts; it does not measure training-run variation because the fitted model is deterministic and only one training split was used.

Both methods shared the same 60-second time cap, but actual compute differed substantially: MPC used 12.747 seconds and 7.68 million model transitions; the direct policy used 0.007 seconds and no model transitions. The result supports no matched-actual-compute advantage claim. On these fixed cases and settings, the direct policy succeeded more often and used less compute.

## Per-seed outcomes

The notebook output retains full precision, step counts, status, and model-call counts for all 40 episodes. This paired summary lists each episode return.

| Seed | MPC success | MPC return | Direct success | Direct return | Direct steps |
| ---: | --- | ---: | --- | ---: | ---: |
| 3000 | No | -20.7910 | Yes | 89.7000 | 103 |
| 3001 | No | -20.9553 | Yes | 89.7000 | 103 |
| 3002 | No | -20.8251 | Yes | 89.5000 | 105 |
| 3003 | No | -20.5797 | Yes | 89.3000 | 107 |
| 3004 | No | -20.8391 | Yes | 89.7000 | 103 |
| 3005 | No | -20.9532 | Yes | 89.4000 | 106 |
| 3006 | No | -20.7992 | Yes | 89.3000 | 107 |
| 3007 | No | -20.8796 | Yes | 89.7000 | 103 |
| 3008 | No | -20.5300 | Yes | 89.7000 | 103 |
| 3009 | No | -20.5981 | Yes | 89.7000 | 103 |
| 3010 | No | -20.9362 | Yes | 89.4000 | 106 |
| 3011 | No | -20.6200 | Yes | 89.7000 | 103 |
| 3012 | No | -20.8497 | Yes | 89.5000 | 105 |
| 3013 | No | -20.8277 | Yes | 89.4000 | 106 |
| 3014 | No | -20.3673 | Yes | 89.7000 | 103 |
| 3015 | No | -20.7858 | Yes | 89.4000 | 106 |
| 3016 | No | -20.4744 | Yes | 89.8000 | 102 |
| 3017 | No | -20.5502 | Yes | 89.7000 | 103 |
| 3018 | No | -20.9376 | Yes | 89.7000 | 103 |
| 3019 | No | -20.6882 | Yes | 89.7000 | 103 |

## Runtime and limits

- Python 3.13.14, NumPy 2.5.4, scikit-learn 1.9.1, Gymnasium 0.29.1.
- Linux WSL2, kernel 6.18.54.1-microsoft-standard-WSL2, x86_64, 24 logical CPUs. The notebook used CPU NumPy and scikit-learn; it did not use a GPU.
- No external data or model weights were used. Peak memory was not measured.
- All notebook cells completed. The 60-second cap passed for both methods.
- This is one deterministic model fit and one small set of simulator starts. The results do not establish a general advantage or disadvantage for learned world models.

The complete intervention and episode records are embedded in [Notebook 14](../../notebooks/14_decision_usefulness.ipynb).
