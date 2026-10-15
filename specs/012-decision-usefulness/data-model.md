# Data Model: Decision-Usefulness Evaluation

## Transition sample

- Training episode seed and step.
- Current position and velocity.
- Applied force.
- Observed next position and velocity.
- Split: training, intervention, or closed-loop evaluation.

Training and held-out splits use disjoint seed groups.

## Intervention case

- Held-out starting state.
- Candidate action from the fixed set {-1, 0, 1}.
- Predicted and observed next state.
- Predicted and observed utility for moving toward the fixed goal.
- State prediction error and action-ranking pair status.

Pairs with equal observed or predicted utility are ties. Exclude them from pairwise agreement and count them.

## Closed-loop episode

- Policy: learned-model MPC or direct momentum rule.
- Evaluation seed and reset state.
- Per-step actions and states.
- Success, return, episode length, elapsed time, model calls, and completion status.

The two policies use identical evaluation seeds, goal, success rule, and maximum episode steps.

## Summary

- Held-out state prediction error.
- Pairwise action-ranking agreement and number of non-tied pairs.
- Per-policy success rate, mean return, and elapsed time.
- Paired per-seed outcome differences and interval method.
- Missing or failed runs and compute-budget status.
