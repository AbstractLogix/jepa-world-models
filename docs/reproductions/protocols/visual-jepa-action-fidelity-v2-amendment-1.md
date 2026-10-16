# Visual JEPA Action Fidelity v2: Pre-Training Amendment 1

**Status**: Frozen before v2 model training
**Date**: 2026-10-16
**Parent protocol**: [Visual JEPA Action Fidelity v2](visual-jepa-action-fidelity-v2.md)

## Reason

The parent protocol names a one-step state gate in the primary validation outcome and applies numeric one-step limits to the supervised visual control. It does not state numeric one-step state limits for JEPA fits. This amendment resolves that omission before any v2 model fit or validation result is read.

## Fixed rule

For each validation model seed, compute position and velocity mean absolute error over all 60 raw next-state predictions from the 20 fixed contexts and actions -1, 0, and +1. Use unclipped state-readout values. A JEPA seed passes this gate only when:

- position MAE is at most 0.02 MountainCar position units; and
- velocity MAE is at most 0.01 MountainCar velocity units.

The supervised visual control uses the same calculation and limits. It must pass this gate in addition to its other parent-protocol gates. A JEPA objective family is eligible only when at least two of its three paired seeds pass every parent and amendment gate. A selected JEPA seed must pass the same one-step state limits on held-out action contexts.

This is an added validation gate. It does not change the objectives, seeds, partitions, model-selection order, multi-step limits, or planning rules in the parent protocol. Smoke outputs remain ineligible for selection.
