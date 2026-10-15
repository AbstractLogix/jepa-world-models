# Data Model: H-JEPA Hierarchical Planning Tutorial

The lesson uses a fixed, synthetic corridor example. These objects are teaching values only.

| Entity | Fields | Meaning |
| --- | --- | --- |
| Toy observation | room, position, fast visual detail | One synthetic observation description. |
| Level view | level name, illustrative horizon, selected state fields | A hand-authored summary used to explain different abstraction levels. |
| Subgoal step | level, current summary, requested summary, selected next summary | One row in the top-down planning chain. |
| Primitive action list | short action names | A hand-authored low-level response to the final subgoal. |

No learned latent, action-conditioned predictor, image, checkpoint, or environment state is stored.
