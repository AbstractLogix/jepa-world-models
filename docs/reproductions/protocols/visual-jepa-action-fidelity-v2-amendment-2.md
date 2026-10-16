# Visual JEPA Action Fidelity v2: Pre-Training Amendment 2

**Status**: Frozen before v2 model training
**Date**: 2026-10-16
**Parent protocol**: [Visual JEPA Action Fidelity v2](visual-jepa-action-fidelity-v2.md)

## Reason

The parent protocol defines representation gates on validation embeddings, while each training pair produces both context and target embeddings. The diagnostic plan requires both distributions. This amendment fixes the gate scope before any v2 model fit or validation result is read.

## Fixed rule

For every JEPA model/data seed pair, report context and target validation embedding finiteness, per-dimension standard deviation, low-spread dimension count, effective rank, and embedding norms. Apply the same representation gate independently to both distributions:

- every value must be finite;
- no more than 16 of 32 dimensions may have standard deviation below 1e-3; and
- effective rank must be at least 8.

The linear state probe remains fit on training context embeddings and their matching state labels. Its validation metrics and the action-fidelity gates remain as defined in the parent protocol and Amendment 1. A model fails representation health if either its context or target validation embeddings fail.

This amendment clarifies which existing model outputs must pass the representation gate. It does not change the thresholds, data, objectives, model capacity, probe, selection order, held-out rules, or compute budget. Smoke outputs remain ineligible for selection.
