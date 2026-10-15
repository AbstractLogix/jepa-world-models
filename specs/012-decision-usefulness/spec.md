# Feature Specification: Decision-Usefulness Evaluation Lesson

**Feature Branch**: `012-decision-usefulness`
**Created**: 2026-10-14
**Status**: Implemented locally; no paper reproduction claimed
**Input**: User direction to deliver the roadmap's future lessons and evaluate whether models support useful decisions.

## User Scenarios & Testing

### User Story 1 - Inspect action interventions (Priority: P1)

A learner wants to know whether a predictive model responds correctly when the action changes while the starting state stays the same.

**Why this priority**: Prediction under recorded actions does not show that a model can predict outcomes for alternative actions.

**Independent Test**: The learner can inspect held-out states, alternative actions from each state, measured prediction errors, and action-ranking agreement.

**Acceptance Scenarios**:

1. **Given** a held-out state and at least three candidate actions, **When** the learner compares predicted and observed next states, **Then** the report preserves the outcomes for every action from the same state.
2. **Given** action utilities calculated from predicted and observed outcomes, **When** the learner reads the summary, **Then** it reports ranking agreement and defines how ties are handled.

### User Story 2 - Compare closed-loop decisions (Priority: P1)

A learner wants to compare a model-based planner with a direct control policy on the same task cases.

**Why this priority**: A model's usefulness depends on the decisions and outcomes it supports, not only on prediction scores.

**Independent Test**: The learner can inspect paired episode outcomes for the planner and direct policy, the shared action and compute limits, and the measured compute for each method.

**Acceptance Scenarios**:

1. **Given** a fixed set of evaluation starts, **When** both methods run, **Then** each uses the same starts, task definition, episode limit, and stated compute budget.
2. **Given** the completed episodes, **When** the learner reads the report, **Then** it shows every result, success and return summaries, paired differences, and missing runs.
3. **Given** model-based and direct-policy results, **When** their compute use differs, **Then** the report states the difference and does not claim a matched-budget advantage.

### Edge Cases

- A model may fail or return an invalid state for an intervention. Preserve the failure and do not silently drop that action.
- Two actions may have equal utility. Exclude tied pairs from ranking agreement and report the number of pairs excluded.
- A planned episode may time out or fail to reach the goal. Keep it in the outcome summary.
- A fitted model may perform well on recorded actions and poorly on changed actions. Report both measurements separately.
- If either method exceeds its compute limit or has missing evaluation episodes, label the comparison partial.

## Requirements

### Functional Requirements

- **FR-001**: The lesson MUST label its local state-space model as an illustrative analogue, not a JEPA model or a paper reproduction.
- **FR-002**: Before final evaluation, the lesson MUST define the task, training and held-out data, intervention actions, direct policy, success rule, episode limit, seeds, and compute budget.
- **FR-003**: The lesson MUST evaluate at least three alternative actions from each of at least 20 held-out states and preserve each predicted and observed outcome.
- **FR-004**: The lesson MUST report held-out prediction error and action-ranking agreement, including the number of non-tied action pairs.
- **FR-005**: The lesson MUST compare the model-based planner and direct policy on at least 20 paired evaluation starts with the same task and action-step limit.
- **FR-006**: The lesson MUST report each episode outcome, success rate, return summary, paired outcome differences, compute budget, actual compute, and any missing or failed runs.
- **FR-007**: The report MUST separate local measurements from paper claims and MUST NOT claim a model advantage when the methods do not share the stated compute budget.
- **FR-008**: The lesson MUST add no dependency to the project's environment files and MUST require no downloaded dataset, checkpoint, or robot hardware.

### Key Entities

- **Intervention case**: One held-out starting state evaluated under multiple actions, with predicted and observed next states and utilities.
- **Evaluation episode**: One policy run from a fixed start, with success, return, step count, runtime, and completion status.
- **Compute budget**: The shared limit for evaluation time or model operations, plus the actual amount used by each method.
- **Local evidence report**: A dated record of the protocol, raw outcomes, summaries, sources, and claim limits.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The lesson reports at least 60 action-intervention outcomes from at least 20 held-out starting states, with three or more actions per state.
- **SC-002**: The lesson reports at least 20 paired closed-loop outcomes for the planner and direct policy on identical evaluation starts.
- **SC-003**: The report includes prediction error, action-ranking agreement, success and return summaries, paired differences, compute limits, and actual compute for both methods.
- **SC-004**: Every planned run is present or is listed as missing with its reason; incomplete runs are labeled partial.
- **SC-005**: The notebook runs locally without external data or weights, and the report limits claims to the measured simulator and protocol.

## Assumptions

- The lesson uses the existing MountainCar continuous-control simulator as a local state-space task.
- A small fitted transition model and a simple direct control rule are sufficient to teach evidence levels; this is not a benchmark claim about JEPA.
- The project evaluation protocol supplies the reporting rules for paired cases, variation, and compute.
- The learner has the Python and notebook environment described in the project quickstart.
