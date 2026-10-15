# Data Model: Sampling-Based Planning Tutorial

## Toy task

- `state`: planar position and velocity `(x, y, vx, vy)` with bounded coordinates and speed.
- `obstacle`: one closed rectangular region between the start and goal; a transition is invalid if its swept segment intersects the obstacle.
- `start`: a fixed, seeded point and low velocity in a declared start region.
- `goal`: a target region beyond the obstacle; success is position within the declared radius with bounded speed.
- `episode_limit`: a fixed maximum count of executed environment steps.

## Toy dynamics and actions

- `action`: one of nine bounded planar accelerations from the Cartesian product `{-a, 0, a} × {-a, 0, a}`.
- `environment_step`: update velocity and position with a fixed discrete-time point-mass rule, then enforce declared speed and world bounds.
- `valid_transition`: reject an action when the swept position segment crosses the obstacle or leaves the world bounds.
- The environment and the planner use the same hand-authored transition function. It is not a learned model.

## Planners and budget

- `RRT`: select the tree state nearest a sampled target (with a declared goal bias), then expand with one sampled action.
- `KPIECE`: map tree positions into a fixed grid, prefer lower-coverage cells, then expand a selected state with one sampled action.
- `CEM`: sample complete action sequences from categorical per-step distributions, score terminal goal distance and invalid paths, and update the distributions from elites.
- `transition_budget`: a fixed maximum number of one-step planner-model evaluations per decision, shared by all three planners. Record actual calls, including early termination and rejected proposals.
- `replan`: discard each planner's temporary search state after executing the same fixed action prefix; plan again from the observed toy state.

## Evaluation records

- `episode`: planner, group seed, paired start id, success, executed steps, planner transition calls, runtime, and terminal state.
- `evaluation_group`: one of five declared groups, each containing 20 shared starts for all planners. It captures evaluation-case and planner-sampling variation, not training variation.
- `report`: protocol version, task and dynamics, planner settings, CPU/software versions, planned and completed episode counts, missing/error counts, per-group summaries, aggregate outcomes, and claim status.

The notebook regenerates all records from fixed seeds in memory. The raw episode records are not committed.
