# Data Model: Persist Raw Evaluation Records

Each CSV row represents one method on one shared start. All rows remain in the file, including unsuccessful episodes and captured errors.

## Notebook 20: `policy-noise-episodes.csv`

**Expected rows**: 300 (5 evaluation groups × 20 starts × 3 methods).

**Unique key**: `(evaluation_seed, start_id, method)`.

| Field | Meaning |
| --- | --- |
| `method` | `raw_action`, `policy_noise`, or `direct_control`. |
| `evaluation_seed` | Evaluation group seed. |
| `start_id` | Start index within the group. |
| `start` | Shared initial position for the paired methods. |
| `success` | Whether the episode reached the fixed success threshold. |
| `steps` | Executed environment steps. |
| `model_calls` | Toy forward-model calls made by the planner. |
| `final_position` | Position after termination. |
| `mean_selected_action_error` | Mean predicted-versus-actual error for selected planner actions; blank for direct control. |
| `off_support_fraction` | Fraction of selected planner actions outside the toy policy support; blank for direct control. |

## Notebook 21: `sampling-based-episodes.csv`

**Expected rows**: 300 (5 evaluation groups × 20 starts × 3 methods).

**Unique key**: `(group_seed, paired_start, method)`.

| Field | Meaning |
| --- | --- |
| `method` | `RRT`, `KPIECE`, or `CEM`. |
| `group` | One-based evaluation group index. |
| `group_seed` | Seed used to generate the shared start and planner seed. |
| `paired_start` | Stable identifier for the shared start. |
| `start_x`, `start_y` | Initial position. |
| `success` | Whether the episode reached the goal. |
| `status` | `success`, `episode_limit`, `invalid_planned_action`, `no_executable_plan`, or a captured error status. |
| `steps` | Executed actions. |
| `transition_calls` | One-step transition evaluations used by the planner. |
| `runtime_seconds` | Measured episode runtime; varies across executions. |
| `final_x`, `final_y` | Final position. |
