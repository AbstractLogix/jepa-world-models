# Feature Specification: Planning Limits Lesson

**Feature Branch**: `004-planning-limits`

**Created**: 2026-10-14

**Status**: Ready for planning

**Input**: Project vision task T011: compare open-loop plans, feedback MPC, and subgoals by goal distance and rollout horizon.

## User Scenarios & Testing

### User Story 1 - Compare planning methods (Priority: P1)

A learner wants to see when a short imagined rollout can guide an action and when a nearby target helps. They run one notebook in a small, repeatable control task and compare open-loop imagination, feedback MPC, and subgoal MPC.

**Why this priority**: This is the main learning outcome of the lesson.

**Independent Test**: Run the notebook from a clean project environment on CPU. It evaluates the same fixed start seeds for each method, goal distance, and rollout horizon. It reports success for every condition.

**Acceptance Scenarios**:

1. **Given** fixed initial-state seeds, **When** the learner runs each planner at the same goal positions and rollout horizons, **Then** each condition uses the same 20 starts and reports success, steps, and planner calls.
2. **Given** the learner compares methods, **When** the result table and plots appear, **Then** paper values and local measurements are shown in separate sections.

### User Story 2 - Understand feedback and subgoal limits (Priority: P2)

A learner wants to understand why a fresh observation may not fix a short-horizon plan and why a nearby subgoal may help.

**Why this priority**: The paper separates rollout reach, feedback, and subgoal effects. The lesson must teach these as different choices.

**Independent Test**: Inspect the method descriptions and one plotted comparison. Confirm that pure imagination uses its own predicted state, MPC uses the new environment state, and subgoal MPC changes the target while keeping the same rollout horizon.

**Acceptance Scenarios**:

1. **Given** a short rollout and a distant goal, **When** the learner reads the comparison, **Then** the notebook explains that an exact dynamics model in this simple environment does not test visual-model accuracy or robot control.
2. **Given** the subgoal method, **When** it runs, **Then** it uses a five-step-ahead target from a same-seed momentum-reference rollout and labels that scripted reference as a teaching aid, not an expert demonstration.

### User Story 3 - Interpret evidence honestly (Priority: P3)

A learner wants to know which results come from the paper and which come from this tutorial.

**Why this priority**: The local environment is much smaller than the paper's Meta-World evaluation.

**Independent Test**: Read the notebook introduction and result summary. Confirm that the local experiment is marked illustrative and does not claim to reproduce the paper's success rates.

**Acceptance Scenarios**:

1. **Given** the notebook reports paper values, **When** it cites the source, **Then** it includes the paper's task and episode context.
2. **Given** the notebook reports local values, **When** it labels them, **Then** it states the environment, starts, seeds, model, and planner settings.

### Edge Cases

- A planner does not reach a subgoal within the episode limit.
- A start state is already at or past the local target.
- CEM candidates converge to the same action sequence or fail to reach the goal.
- A local model transition differs from the installed Gymnasium transition.
- The result differs across Gymnasium versions.
- An evaluation condition has missing episodes or a planner error.

## Requirements

### Functional Requirements

- **FR-001**: The notebook MUST explain the paper's distinction between imagined rollout horizon and goal lookahead, and link to the primary paper source.
- **FR-002**: The notebook MUST use the `MountainCarContinuous-v0` state and action spaces and MUST verify its local transition function against Gymnasium transitions before planning.
- **FR-003**: The notebook MUST compare pure imagination, feedback MPC, and feedback MPC with five-step-ahead targets from a same-seed momentum-reference rollout. All planning methods MUST use the same action predictor and planner budget for a given condition.
- **FR-004**: The notebook MUST vary the goal position across `0.0`, `0.25`, and `0.45`, and the planning horizon across 5 and 20 environment steps.
- **FR-005**: The notebook MUST evaluate 20 paired initial-state seeds per condition with a maximum of 300 environment steps per episode. Success MUST mean position at or beyond the condition goal with non-negative velocity. Report success rate, Wilson 95% interval, steps to success, and planning calls. Failed and missing runs MUST remain visible.
- **FR-006**: The notebook MUST include a simple momentum controller as a baseline and MUST state its action rule. The same rule MUST provide the reference trajectory used by the subgoal method.
- **FR-007**: The notebook MUST identify the Gymnasium task as a small state-space teaching analogue. It MUST NOT claim to reproduce the paper's Meta-World or BridgeData results.
- **FR-008**: The notebook MUST state the paper-reported values separately from local results. It MUST record Python, Gymnasium, CPU/GPU, random seeds, and planner settings.
- **FR-009**: The notebook MUST run without external data, model checkpoints, or robot hardware. It MUST use the project's existing dependencies only.

### Key Entities

- **Planning condition**: Planner mode, goal position, rollout horizon, CEM budget, and initial-state seed set.
- **Evaluation episode**: Start state, goal, action sequence, success status, steps, and planner call count.
- **Paper result**: A cited value with the paper's environment, task set, model, and episode conditions.
- **Local result**: A measurement from this notebook with its environment, version, seeds, and planner settings.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The notebook runs from top to bottom on CPU without downloads or robot access.
- **SC-002**: Every planned condition reports the same 20 fixed start seeds for the three planning methods, with all failures and missing runs visible.
- **SC-003**: Results show success by goal position and rollout horizon for pure imagination, feedback MPC, subgoal MPC, and the momentum baseline.
- **SC-004**: The notebook keeps paper-reported and local measurements separate and labels the local experiment illustrative.
- **SC-005**: The transition check matches Gymnasium for the tested states and actions within absolute tolerance `1e-6`.

## Assumptions

- Use Gymnasium's deterministic continuous MountainCar task as a small control example. It is not Meta-World, a visual task, or a robot task.
- Use an exact local dynamics function for planning and verify it against the installed environment. This isolates planning choices from learned-model prediction error.
- Pure imagination updates its internal state only with its own predicted transitions. Feedback MPC replans from the latest environment observation. Subgoal MPC uses the same feedback loop and planner but targets the state five steps ahead on a same-seed momentum-reference path.
- Use CEM with 64 action-sequence candidates, 8 elites, and 2 iterations per planning call. The subgoal method uses the momentum controller's same-seed reference state five steps ahead. Record any adjustment needed to keep the notebook usable.
- The cited paper version is arXiv v1, submitted 2026-09-30. It reports results on sixteen Meta-World tasks with twenty paired episodes per task; the local experiment uses different conditions.
- No author code or benchmark dataset is used by this teaching lesson.
