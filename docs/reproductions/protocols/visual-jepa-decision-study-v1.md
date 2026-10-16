# Preregistration: Visual JEPA Decision Study v1

**Status**: Preregistered before development planner tuning.<br>
**Date**: 2026-10-15<br>
**Study**: [Feature 029](../../../specs/029-matched-visual-jepa-study/spec.md)

## Research question and hypothesis

**Question**: Does a learned action-conditioned visual representation improve task decisions beyond a direct momentum controller and beyond the effect of planner initialization?

**Primary hypothesis**: With the same locked CEM budget and momentum-prior initialization, JEPA dynamics will improve paired episode return over the direct visual-momentum controller and will show positive incremental return over the same prior with non-JEPA dynamics. This is a local hypothesis, not a general JEPA claim.

**Primary comparison/outcome**: Mean paired MountainCar episode-return difference between prior-guided JEPA CEM and the direct visual-momentum controller on the same 20 final starts. Report each JEPA training-seed difference and a seed-aware 95% hierarchical bootstrap interval. Do not report a significance claim from this small study.

**Secondary outcomes**: Success, steps to goal, return, uniform-versus-prior initialization delta within each dynamics model, prediction/intervention fidelity, common task-outcome action ranking, probe performance, model calls, candidate rollout transitions, planning time, and full elapsed time.

## Task and rules

- Environment: `MountainCarContinuous-v0`, Gymnasium 0.29.1-compatible task.
- Goal position: 0.45. Action range: [-1, 1]. Episode cap: 300 steps.
- The synthetic frame generator and visual momentum feature controller are reused from Notebook 22.
- Success is Gymnasium termination at the goal. Timeouts remain in the denominator. Return is the actual environment return, including action cost and goal reward.
- The direct visual-momentum controller is handcrafted. Oracle and state-input dynamics are privileged. Visual models receive rendered frames; their linear state probe is fit on training data and selected on validation data only.

## Data separation and seeds

| Role | Seeds / records | Use |
|---|---|---|
| Planner tuning | 8200–8209 | Select settings from the predeclared development grid only. |
| Planner confirmation | 8210–8219 | Confirm the selected exact planner once; no retuning. |
| Training data | 8300–8304 | Five shared data groups. Twenty random-action episodes per group; episodes 0–15 train, 16–19 validate. |
| Model training | 7401–7405 | Five independent training seeds paired by index with data seeds 8300–8304. |
| Representation test | 8400–8404 | Independent trajectories for probe and intervention-fidelity diagnostics only. |
| Intervention states | 8600–8604 at steps 40, 90, 140, 190 | Twenty held-out starts, each paired with actions -1, 0, and 1. |
| Final evaluation | 8700–8719 | Twenty shared starts. Exact planner gate runs first. Learned models run only after a pass. |
| Smoke | 8900–8901 | CPU-only pipeline checks; never used for final claims. |

No development, validation, smoke, intervention, or evaluation seed is used to fit model parameters. The representation test set is not used to select probes or planner settings.

## Planner development search

The final planner is not selected from held-out model outcomes. Development tuning has a maximum of 500,000,000 candidate transition evaluations.

**Stage A** uses the 10 tuning starts, 24 candidates, 4 elites, 2 updates, and replans every 5 environment steps. It compares horizons `{10, 60, 120, 180}`, initialization `{uniform, momentum prior}`, and rollout score `{historical terminal potential, cumulative Gymnasium return}`. The action sequence is represented by one action per five environment steps. The historical potential is `-(position - goal)^2 - 0.2 * velocity^2` at the rollout endpoint. The cumulative score uses the actual Gymnasium reward and does not add extra shaping.

**Stage B** uses the selected Stage A horizon, initialization, and score on the first five tuning starts. It compares candidate counts `{24, 48}` and update counts `{2, 4}`. Elites are `ceil(candidates / 6)`. All other settings remain fixed.

**Selection rule**: Choose by success count, then mean environment return, then fewer candidate transition evaluations, then shorter horizon. If still tied, use the uniform initialization and cumulative Gymnasium return. Evaluate the selected configuration once on all 10 confirmation starts. Lock it only if it succeeds on at least 9/10 confirmation starts and at least 18/20 development starts overall. Keep every attempted setting in the development CSV.

No planner hyperparameter or score is changed after the development confirmation set is read.

## Held-out oracle gate and final matrix

After a successful development lock, evaluate the locked exact-dynamics planner on final starts 8700–8719. Do not train/evaluate learned methods until this oracle gate is at least 18/20 successful. A failed final oracle gate closes the learned-model evaluation and leaves the result as a planner limitation.

If the oracle gate passes, compare:

1. Exact-dynamics CEM, uniform and momentum-prior initialization (privileged).
2. Learned state-space CEM, uniform and momentum-prior initialization (privileged; five training seeds).
3. Learned visual reconstruction dynamics, uniform and momentum-prior initialization (visual; five seeds).
4. Action-conditioned visual JEPA dynamics, uniform and momentum-prior initialization (visual; five seeds).
5. Direct visual-momentum controller (visual; one outcome per start).

The momentum-prior initialization uses the same fixed hand-written visual-momentum rule for each model. The planner settings, action bounds, replanning interval, and rollout budget match across CEM conditions. Actual compute time may differ by method and device. Report that difference; do not claim compute parity.

## Common outcome and diagnostics

- Score candidate rollouts with the same task reward adapter. State dynamics produce position/velocity directly; visual latent dynamics use a linear readout trained on training data and selected only on validation data.
- Fit the position/velocity readout after freezing each visual model. Report test MAE and R² separately from planning outcomes.
- For each intervention, evaluate each candidate action with a common post-action continuation under the fixed momentum controller. Compare predicted and observed continuation returns for action ranking. Label this as an evaluator-side simulator diagnostic, not a deployable reward model.
- Report model-native losses in their own units. Never compare raw pixel, latent, and state distances as one metric.
- Report embedding spread, position/velocity probe scores, and action sensitivity as representation diagnostics only.

## Failure and abstention rules

- If the development oracle fails its lock rule, do not run the final oracle gate or learned evaluation.
- If the final oracle gate fails, retain its rows and do not run learned evaluation.
- If a model collapses, returns non-finite values, misses a seed, or exceeds the compute ceiling, preserve all rows and mark the comparison partial. Do not replace a seed or reduce episode counts after viewing results.
- If a visual probe is not valid on training/validation evidence, do not open that method's planner evaluation.
- CPU smoke and small device probes do not satisfy the comparison threshold.

## Compute ceiling and analysis

- Planner tuning ceiling: 500 million candidate transition evaluations, including all attempted development settings.
- Final planner ceiling: 750 million candidate transition evaluations. Stop before opening any additional case if the estimated remaining budget exceeds this limit.
- Training: five model seeds per trained family, 35 epochs, shared generated data. Record actual training seconds and peak allocated GPU memory.
- Report method-level success with binomial intervals for fixed-start controls, per-model-seed outcomes for trained models, and paired return differences. Use a two-level bootstrap that resamples training seeds and then shared evaluation starts. Show all five seed points. Treat intervals as descriptive; do not claim generalization or statistical significance.

## Claim limits

This is a local, small-scale synthetic MountainCar experiment. It is not a V-JEPA 2, LeJEPA, VISReg, or other paper reproduction. It cannot establish robotics capability, general JEPA superiority, or independent replication. Negative and gated outcomes remain publishable evidence.

## Preregistration integrity

This file defines the search and analysis before development tuning. A separate protocol-lock JSON records the selected settings, code hash, and development outcomes. Final mode must verify both the preregistration hash and the lock/gate identity. The preregistration text is not edited in response to development or held-out outcomes.
