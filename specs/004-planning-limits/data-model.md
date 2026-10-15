# Data Model: Planning Limits Lesson

## Planning condition

- `mode`: pure imagination, feedback MPC, momentum-reference subgoal MPC, or momentum baseline.
- `goal_position`: one of 0.0, 0.25, or 0.45.
- `rollout_horizon`: 5 or 20 environment steps for planner modes.
- `seed`: one of the fixed 20 paired reset seeds.
- `episode_limit`: maximum of 300 environment steps.
- `planner_settings`: CEM candidate count, elite count, iteration count, and episode limit.
- `reference_path`: optional same-seed trajectory from the momentum rule; subgoal mode targets the state five steps ahead on this path.
- `gymnasium_version`: installed package version used for the run.

## Evaluation episode

- `condition`: planning condition above.
- `initial_state`: position and velocity returned by seeded reset.
- `success`: whether the state reached the condition goal before the episode limit.
- `steps`: actions executed, including failed episodes up to the limit.
- `planning_calls`: number of CEM planning calls; zero for the momentum baseline.
- `final_state`: position and velocity at success or limit.
- `error`: optional failure description; failed or missing runs remain visible.

Success means position is at or above the condition goal and velocity is non-negative.

## Local summary

- `episodes`: all episode records for a condition.
- `success_rate`: successful episodes divided by planned episodes.
- `wilson_95`: Wilson interval for the success rate.
- `mean_steps_to_success`: mean over successful episodes; show successful count beside it.
- `mean_planning_calls`: mean over all planned episodes.

## Paper result

- `source`: arXiv version 1 link.
- `setting`: sixteen Meta-World tasks, twenty paired episodes per task, V-JEPA 2.1, and stated rollout/feedback/subgoal configuration.
- `reported_success`: value copied from the paper with the relevant condition.

Paper results and local summaries are separate records and must not be pooled.
