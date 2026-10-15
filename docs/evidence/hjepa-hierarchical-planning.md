# H-JEPA hierarchical planning tutorial evidence

**Status**: Complete for the local teaching analogy; no H-JEPA paper result is claimed.
**Notebook**: [`19_hjepa_hierarchical_planning.ipynb`](../../notebooks/19_hjepa_hierarchical_planning.ipynb)
**Run date**: 2026-10-15

## Local run

- Executed all 7 cells in a fresh Jupyter kernel from the project `.venv`.
- Kernel Python: 3.13.14. The notebook uses only the Python standard library.
- Both code cells completed without errors.
- `nbformat` validation passed. Both cells have stable IDs and execution counts.

## Observed output

- Two toy observations share the same room and position but use different visual-flicker values.
- The hand-authored level-1 summary preserves the difference. The hand-authored level-2 and level-3 summaries omit it.
- The top-down example passes `goal region` to `hall region`, then `doorway`, then `move`, `turn`, `move`. Assertions confirm that each subgoal is the next level's goal.

These are outputs of the toy rules in the notebook. They do not show that trained H-JEPA representations discard visual detail or that hierarchical planning improves task performance.

## Evidence boundary

The [H-JEPA v1 paper](https://arxiv.org/abs/2610.06805v1) reports 73.3% ± 3.5 success for three-level planning and 18.0% ± 3.5 for flat LeWM on Visual AntMaze across three model seeds. These are paper values. The official local evaluator has not produced a task result; see the [AntMaze preflight](../reproductions/h-jepa-antmaze-preflight.md). Notebook 19 does not load paper assets, implement H-JEPA, run AntMaze, or reproduce the reported success rates.
