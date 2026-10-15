# Research: Decision-Usefulness Evaluation Lesson

**Reviewed**: 2026-10-14

## Findings

### Decision-making evaluation should measure interventions and executed outcomes

Yu et al., *How Should World Models Be Evaluated? A Decision-Making-Centric Position* (arXiv:2606.15032), argues that evaluation should match the model's claimed use. For embodied decision-making, it discusses counterfactual action fidelity, closed-loop validity, policy-ranking agreement, optimization lift, model exploitability, and uncertainty calibration. It organizes evidence from artifact diagnostics through policy-optimization utility. The paper presents an evaluation framework; it does not define one universal metric.

### Direct-policy comparisons remain uncommon

Jena et al., *Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence* (arXiv:2609.29669), reviews 160 benchmarks and reports 11 with an explicit world-model versus direct-policy comparison. It also notes that few benchmarks connect prediction to executed action. This supports reporting a direct-policy baseline and closed-loop outcomes as separate evidence.

### Existing lesson boundary

Notebook 10 measures planning horizon and feedback in MountainCar using exact dynamics and a momentum baseline. It does not fit a model from data or test alternative actions from the same held-out state. The new lesson uses MountainCar to add those measurements without changing Notebook 10's existing results.

## Decisions

- Build a local state-space analogue, not a JEPA implementation or paper reproduction.
- Fit a deterministic polynomial transition model on simulator-generated training data. Use separate seed groups for training, action interventions, and closed-loop evaluation.
- Evaluate three force choices at each held-out state and report next-state prediction error plus pairwise action-utility ranking agreement. Exclude and count ties.
- Compare learned-model MPC with Notebook 10's direct momentum rule on 20 paired evaluation starts. Use one fixed 300-step episode limit and the same 60-second elapsed-time cap for each policy evaluation.
- Record actual model-call counts and elapsed time. A local outcome cannot support a matched-budget advantage when the methods exceed or do not share the stated cap.
- Keep the existing project stack. Do not download data, checkpoints, or simulator assets.

## Alternatives Considered

- **Reproduce the papers' robot benchmarks**: out of scope for this local lesson because they need robot tasks, released models, or evaluation infrastructure not used by this project.
- **Use exact simulator dynamics**: rejected for the new measurement because an exact model would not test learned counterfactual predictions.
- **Use only one-step prediction error**: rejected because it does not show whether changing an action changes a decision-relevant prediction or whether a planner succeeds in closed loop.
- **Add a new evaluation framework dependency**: rejected because the experiment can use the existing Python stack.

## Sources

- Yu et al., [How Should World Models Be Evaluated? A Decision-Making-Centric Position](https://arxiv.org/abs/2606.15032).
- Jena et al., [Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence](https://arxiv.org/abs/2609.29669).
- [Existing MountainCar planning lesson](../../notebooks/10_planning_limits.ipynb) and its [local run report](../../docs/reproductions/planning-limits.md).
