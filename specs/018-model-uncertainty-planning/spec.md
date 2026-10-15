# Feature Specification: Calibrate and Stress-Test a World Model

**Feature Branch**: `[018-model-uncertainty-planning]`  
**Created**: 2026-10-15  
**Status**: Implemented; local execution complete; not a paper reproduction
**Input**: Close the project evaluation gap for model uncertainty and exploitability with one CPU-sized local tutorial.

## User Scenarios & Testing

### User Story 1 - Check uncertainty under distribution shift (Priority: P1)

As a learner, I want to compare a dynamics ensemble's predicted uncertainty with its actual prediction error on random-action and planner-generated states so that I can see when confidence stops matching accuracy.

**Why this priority**: A model that supports planning must be checked on the states induced by planning, not only on random training-like transitions.

**Independent Test**: Execute the calibration section of Notebook 17. It must use disjoint training, calibration, in-distribution test, and policy-shift test seeds, and report interval coverage and width for each split.

**Acceptance Scenarios**:

1. **Given** training and calibration episodes, **When** the ensemble interval is calibrated on validation data, **Then** the notebook reports its target coverage and scale separately from results on held-out random-action and policy-shift transitions.
2. **Given** policy-shift test transitions, **When** interval coverage is calculated, **Then** each transition is compared with the simulator's observed next state and the notebook reports undercoverage or overcoverage without retuning on test data.

### User Story 2 - Measure planner exploitability (Priority: P2)

As a learner, I want to compare model-optimized plans with their outcomes in the simulator so that I can measure whether a planner selects plans that exploit model errors.

**Why this priority**: High predicted return does not prove that a plan works in the environment.

**Independent Test**: Execute the planning section on the same held-out starts for mean-ensemble CEM, uncertainty-penalized CEM, and a direct momentum controller. The notebook must retain predicted and actual plan returns, paired episode outcomes, and compute.

**Acceptance Scenarios**:

1. **Given** the same start state and candidate budget, **When** each CEM score is optimized, **Then** the top plan's predicted return and its same-sequence simulator return are recorded as separate values.
2. **Given** five independent model-training runs and 20 paired evaluation starts for each learned-planner method, **When** local results are summarized, **Then** the notebook reports every run, success, return, compute, and uncertainty interval, and labels this state-based MountainCar experiment as an analogue rather than a JEPA or paper reproduction.

## Requirements

### Functional Requirements

- **FR-001**: Notebook 17 MUST use `MountainCarContinuous-v0` with a state-based action-conditioned dynamics model and MUST label the model as a local analogue, not JEPA.
- **FR-002**: Training, calibration, in-distribution testing, and policy-shift testing MUST use distinct recorded seeds.
- **FR-003**: The notebook MUST fit five independent ensemble runs and calibrate the 90% one-step prediction interval using validation data only.
- **FR-004**: The notebook MUST report coverage and interval width separately for in-distribution and planner-induced policy-shift transitions; it MUST NOT tune on either test split.
- **FR-005**: The notebook MUST compare mean-ensemble CEM, uncertainty-penalized CEM, and direct momentum control on identical evaluation starts and stated planner budgets.
- **FR-006**: For each initial selected open-loop plan at a final-evaluation start, the notebook MUST record model-predicted return, same-sequence simulator return, and their difference as a selected-plan gap. The notebook MUST NOT label this 8-step local diagnostic as the paper's full policy-level XGap.
- **FR-007**: The notebook MUST retain each model-run seed, training sample count and time, plus per-episode status, success, return, steps, model calls, wall time, and missing or failed runs.
- **FR-008**: The report MUST include mean, median, and a stated 95% interval across independent training runs; paired per-start differences MUST be reported for methods evaluated on the same starts.
- **FR-009**: The lesson MUST distinguish prediction calibration, model-optimized plan gap, closed-loop task success, and direct-policy comparison as separate evidence.
- **FR-010**: README, roadmap, curriculum guide, coverage map, reproduction matrix, project alignment, and remaining-work records MUST label the result as local and illustrative, not a paper reproduction.

### Key Entities

- **Training run**: One independently seeded ensemble fit using simulator data from its recorded training episodes.
- **Calibration case**: One held-out state-action transition used to scale ensemble spread into a 90% prediction interval.
- **Evaluation case**: One fixed transition or reset seed reserved for in-distribution, policy-shift, or closed-loop testing.
- **Plan record**: One initial state, selected action sequence, predicted return, simulator return, and exploitability gap.
- **Episode record**: One method, model-training seed, evaluation seed, status, task outcome, and compute record.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Notebook 17 runs from a clean project environment on CPU without new data downloads or new dependencies.
- **SC-002**: All five independent model runs have results on the same 20 closed-loop evaluation starts for both learned-planner methods; the direct baseline has one outcome for each of the same starts. No failed or missing episode is omitted.
- **SC-003**: The report gives separately calibrated interval coverage and width for in-distribution and policy-shift transitions, using no test data for calibration.
- **SC-004**: The notebook records predicted and actual return for each initial selected plan at all final-evaluation starts and calculates the signed gap as predicted return minus simulator return.
- **SC-005**: Project documentation records the lesson, run status, exact settings, and claim limits, and does not present it as a JEPA or paper reproduction.

## Assumptions

- The lesson uses a small polynomial-regression ensemble on simulator state. It does not use images, pretrained encoders, robot data, or a JEPA model.
- Five independent fits and 20 paired evaluation starts per fit meet the project's default teaching-comparison scale; intervals remain uncertain with five model runs.
- The uncertainty penalty is fixed before final evaluation. No test result is used to tune it.
- A positive exploitability gap means the model overpredicts the return of the selected plan; a negative gap means it underpredicts that plan.
- This is an educational analogue of evaluation dimensions in current world-model research, not a reproduction of the cited research experiments.
