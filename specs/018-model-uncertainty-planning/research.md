# Research: Model Uncertainty and Planner Exploitability

**Reviewed**: 2026-10-15

## Decision

Use a bounded, state-based MountainCarContinuous experiment to teach two separate checks: whether a model's prediction intervals match held-out errors, and whether a planner's top predicted plan succeeds when executed in the simulator. This is a local analogue. It does not reproduce any cited paper or use a JEPA encoder.

The lesson will use a five-member bootstrap ensemble of polynomial ridge dynamics models. Each of five independent model runs uses separately seeded simulator data. A 90% one-step state interval is scaled on a held-out calibration split, then assessed on independent random-action and planner-induced transitions. The planner uses ensemble mean return or a fixed mean-minus-one-standard-deviation score. The uncertainty penalty is pre-registered; it is not tuned on final test outcomes.

## Primary research basis

### Decision-centric evaluation

Yu et al.'s [decision-making-centric position paper](https://arxiv.org/html/2606.15032) argues that prediction quality alone does not establish decision value. It recommends evaluating model uncertainty, policy-induced distribution shift, and model exploitability. Its exploitability gap compares the model's predicted return for a model-optimized policy with the environment's return for that same policy. It also recommends searching for plans that score well in the model and checking a safe subset in the environment. This feature adapts those evaluation ideas to a small simulator; it does not reproduce the paper's benchmark protocol.

The paper's numeric thresholds are explicitly presented as illustrative and domain-specific. This feature does not reuse them as acceptance thresholds.

### Environment probing is related, but distinct

Song and Cai's [budgeted environment-probing preprint](https://arxiv.org/html/2606.31422) studies language agents that maintain structured beliefs and spend interaction steps to query the environment. It finds a trade-off between repairing beliefs and advancing the task. This is a related calibration direction, but it is not an action-conditioned dynamics model or a visual-control experiment. It is included as a nearby research thread, not as the method implemented here.

### Local environment

The project's lock pins Gymnasium 0.29.1. The official [0.29.1 environment implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/classic_control/continuous_mountain_car.py) defines the two-value state, bounded continuous action, deterministic transitions, reward, and goal. The notebook will cap episodes at 200 steps instead of the environment's default 999-step limit and state this deviation. It will use the environment API rather than copy its transition code.

## Fixed local protocol

- Five independent model runs; each run collects random-action data from its own 10 training episodes and fits a five-member bootstrap ensemble.
- Calibration episodes use seeds 2000–2009. Independent in-distribution test episodes use 3000–3009.
- Policy-shift calibration episodes use 3500–3509 and actions selected by mean-ensemble CEM.
- Final closed-loop evaluation uses the same 20 reset seeds, 4000–4019, for both learned planners and the direct momentum baseline.
- Calibrate a separate position and velocity interval scale from the 90th percentile of absolute standardized residuals on calibration data. Report the resulting coverage and mean width separately for in-distribution and policy-shift transitions.
- Compare mean-return CEM, fixed risk-penalized CEM (`ensemble mean - 1 × ensemble return standard deviation`), and direct momentum control. Fix the planner horizon, candidates, elites, updates, action limits, and episode cap before evaluation.
- For the initial selected 8-step sequence at each final-evaluation start, compute its model-predicted return and its return from executing the same sequence in a fresh simulator at the same state. Define the signed selected-plan gap as predicted minus simulator return. This is not the paper's full policy-level XGap.
- Report every run and episode, mean, median, and a 95% interval across the five model runs. Report paired per-start differences for learned methods against the fixed direct-policy outcomes.

## Alternatives considered

- **Use a pretrained JEPA or image model**: rejected for this lesson because it would add model-weight access and compute gates without being needed to teach the evidence gap.
- **Reuse Notebook 16's controller-prior comparison**: rejected as the main lesson because its single fitted model does not estimate uncertainty or report model-optimized versus realized plan return.
- **Import stable-worldmodel**: not needed for this bounded task. It is an active, broad evaluation framework; this tutorial uses dependencies already declared by the project.
- **Treat model ensemble spread as calibrated probability**: rejected. Ensemble spread is only a score until its interval coverage is measured on held-out cases.

## Limits

The model observes simulator position and velocity, not pixels or JEPA latents. The task is deterministic MountainCarContinuous with a shorter 200-step cap. Ensemble intervals and five-run confidence intervals are uncertain at this scale. Results demonstrate local evaluation methods only; they do not establish a general calibration method, a JEPA finding, or a real-robot outcome.
