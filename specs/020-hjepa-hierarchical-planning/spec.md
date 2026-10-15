# Feature Specification: H-JEPA Hierarchical Planning Tutorial

**Feature Branch**: `[020-hjepa-hierarchical-planning]`
**Created**: 2026-10-15
**Status**: CPU tutorial implemented; official evaluation is separate
**Input**: Project roadmap item for H-JEPA and the user's request to add future tutorials with evidence limits.

## User Scenarios & Testing

### User Story 1 - Understand top-down hierarchical planning (Priority: P1)

As a learner, I want to see how a high-level goal becomes lower-level subgoals so that I can understand how H-JEPA plans across time scales.

**Independent Test**: Run the notebook on CPU. It must show top-down subgoal flow, level-specific latent examples, and an explicit analogy boundary.

**Acceptance Scenarios**:

1. **Given** one visual goal, **When** a learner follows the hierarchy, **Then** the lesson explains that each H-JEPA level predicts farther ahead in its own learned latent space and passes subgoals down to the level below.
2. **Given** the notebook's toy calculation, **When** it removes fast detail at higher levels, **Then** the lesson says that this is hand-authored and does not implement a learned encoder or predictor.
3. **Given** the paper's Visual AntMaze result, **When** the notebook states its value, **Then** it labels the value as paper-reported and links to the paper.

## Requirements

### Functional Requirements

- **FR-001**: The notebook MUST explain action-conditioned prediction at multiple levels, each with its own latent space and a longer prediction horizon at higher levels.
- **FR-002**: The notebook MUST explain top-down planning: a higher-level prediction supplies a subgoal to the planner below it.
- **FR-003**: The local example MUST run on CPU using existing project dependencies and no external data, checkpoints, or hardware.
- **FR-004**: The notebook MUST label its toy state, level mappings, and planner as an educational analogy, not H-JEPA code or paper evidence.
- **FR-005**: Any H-JEPA result in the notebook MUST remain a paper-reported value with a primary-source link.
- **FR-006**: The README, curriculum coverage, roadmap, project research, alignment, remaining work, and project task ledger MUST distinguish the local lesson from the gated Visual AntMaze evaluation.

### Out of Scope

- Downloading or loading H-JEPA checkpoints, AntMaze tasks, or training data.
- Running the official evaluator, training a model, or claiming a paper reproduction.
- Measuring planning compute or comparing hierarchy against a flat planner in the toy example.

## Success Criteria

- **SC-001**: Every notebook code cell runs in a fresh CPU kernel without new dependencies or external assets.
- **SC-002**: The notebook shows one explicit top-down chain from a high-level goal to primitive actions.
- **SC-003**: The notebook states at least three limits of its analogy and makes no local H-JEPA performance claim.
- **SC-004**: Every current project index that lists tutorial coverage labels Notebook 19 as a local analogy and keeps official evaluation status separate.

## Assumptions

- The H-JEPA v1 paper and code revision already pinned in the project remain the sources for this lesson.
- The CPU lesson can explain the planning structure without reproducing the training objective or learned representation.
- Official Visual AntMaze evaluation remains a separate project task and stays gated by the renderer preflight.
