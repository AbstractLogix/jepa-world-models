# Research: H-JEPA Hierarchical Planning Tutorial

**Reviewed**: 2026-10-15

## Decision

Teach the planning structure from the H-JEPA v1 paper. Use a small CPU analogy to show how abstract predictions can pass subgoals down through a hierarchy. Do not present the analogy as an H-JEPA implementation or as evidence for the paper's result.

## Primary Sources

- [H-JEPA v1 paper, arXiv:2610.06805](https://arxiv.org/abs/2610.06805v1): introduces a hierarchy of action-conditioned JEPAs. Each level predicts farther ahead in its own learned latent space. Planning proceeds from the top level toward lower levels as predictions become subgoals.
- [Official H-JEPA code at `97054ecaff83e13839195ec4b92d74c08a0127c5`](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5): pinned implementation and paper reproduction instructions. The repository's README describes a three-level Visual AntMaze result of 73.3% versus 18.0% for flat LeWM.
- [Project preflight](../../docs/reproductions/h-jepa-antmaze-preflight.md): records checkpoint/task hashes and the local EGL/OSMesa failure. No H-JEPA task outcome is available.

## Scope and Limits

- The notebook uses hand-authored state summaries and subgoal rules. It does not train encoders or predictors.
- It does not reproduce the Visual AntMaze tasks, checkpoints, planner, or success-rate comparison.
- The paper's result is a simulated benchmark result. It is not physical-robot validation.
