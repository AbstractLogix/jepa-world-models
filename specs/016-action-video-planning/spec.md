# Feature Specification: Action-Conditioned Video Planning Lesson

**Feature Branch**: `016-action-video-planning`

**Created**: 2026-10-15

**Status**: Implemented; one-fit exploratory simulator analogue; no paper reproduction claim

**Input**: Add a small, reproducible tutorial that connects video observations, action-conditioned prediction, and planning, and reflects current research without claiming to reproduce a paper.

## User Scenarios & Testing

### User Story 1 - Learn planning from visual observations (Priority: P1)

A learner follows rendered simulator observations through feature extraction, action-conditioned prediction, and model-based planning. The learner can compare the planner's imagined outcome with the simulator outcome.

**Why this priority**: The curriculum has video features, latent actions, and state-based planning lessons, but no lesson that connects visual observations to a learned action-conditioned model and planner.

**Independent Test**: Run Notebook 16 from top to bottom on CPU with no checkpoint or dataset download. It reports held-out prediction error and a visualized closed-loop episode.

**Acceptance Scenarios**:

1. **Given** a generated simulator trajectory, **When** the learner extracts visual features and fits the predictor, **Then** held-out prediction error is reported separately from training error.
2. **Given** a held-out start and target, **When** the learner replans from new visual observations, **Then** the notebook records whether the simulator reaches the target.

### User Story 2 - Compare a policy-guided planning prior (Priority: P2)

A learner compares model-based planning with uniform action sampling and with a simple goal-directed action prior, while a direct visual controller provides a policy baseline.

**Why this priority**: Recent work combines a learned policy with latent world-model planning. A small local comparison makes that design choice inspectable without transferring robot datasets or checkpoints.

**Independent Test**: Run both planners on the same held-out starts with matched candidate counts, planning horizon, feedback, and simulator step limit. Report paired outcomes and planner work.

**Acceptance Scenarios**:

1. **Given** identical held-out starts and a fixed planning budget, **When** the two action priors are compared, **Then** the notebook reports per-start success, return, model calls, and elapsed planning time.
2. **Given** local results that differ from a paper, **When** the learner reads the discussion, **Then** the notebook labels the result as a simulator analogue and states why it is not a paper reproduction.

## Edge Cases

- The visual encoder may fail to detect the controlled object or target; the notebook must stop with a clear error rather than use hidden simulator state.
- A planner may exploit prediction errors; held-out one-step interventions and actual simulator outcomes must remain separate measurements.
- A planner may not reach the target within the step limit; report failure and the number of steps instead of dropping the episode.
- A policy prior may perform worse than uniform sampling; this is a valid result and must not change the evaluation set or budget.

## Requirements

### Functional Requirements

- **FR-001**: The lesson MUST use locally generated simulator observations and MUST run without external datasets or model checkpoints.
- **FR-002**: The lesson MUST derive its planning input from rendered observations and MUST NOT pass the simulator's hidden state to the learned predictor or planner.
- **FR-003**: The lesson MUST train an action-conditioned predictor using a documented training split and MUST report its held-out one-step error.
- **FR-004**: The lesson MUST compare uniform-start and policy-prior-start model-based planners under the same action-sample count, horizon, feedback rule, and held-out starts.
- **FR-005**: The lesson MUST compare both planners with a direct visual policy using the same evaluation starts and task-success rule.
- **FR-006**: The lesson MUST report per-method task outcomes, model calls, elapsed planning time, seeds, software versions, and device.
- **FR-007**: The lesson MUST distinguish simulator evidence from paper results and MUST state that the experiment is not a JEPA implementation or paper reproduction.
- **FR-008**: The lesson MUST identify and cite current primary research on action-conditioned latent planning and policy-guided planning, and note relevant released code or asset limits.

### Key Entities

- **Rendered observation**: A simulator frame or short frame history used as the only input to the encoder.
- **Visual feature**: A fixed, documented representation extracted from rendered observations for this small local lesson.
- **Action-conditioned predictor**: A fitted model that predicts the next visual feature from the current feature and action.
- **Action prior**: A distribution or sequence used to initialize candidate action search.
- **Evaluation episode**: A fixed-seed start and target used to compare all control methods.
- **Evidence record**: The run settings, raw episode outcomes, aggregate measures, compute, and claim limits.

## Assumptions

- A compact local simulator analogue is an appropriate first step before a paper-scale robot-data attempt.
- A simple fixed visual feature extractor is acceptable if its limits are explicit and no claim is made that it is a pretrained JEPA representation.
- The local comparison may find no benefit from the policy prior; the lesson remains complete if it reports that result without changing the protocol.
- Reproducing V-JEPA 2-AC or PiJEPA paper results remains a separate task requiring paper-aligned models, data, compute, and evaluation.

## Success Criteria

- **SC-001**: A clean local run completes on CPU without downloading a dataset or checkpoint.
- **SC-002**: The notebook reports training and held-out action-conditioned prediction error on disjoint simulator seeds.
- **SC-003**: The notebook reports paired closed-loop outcomes for both planners and the direct visual policy on the same evaluation starts.
- **SC-004**: The notebook reports planning calls and elapsed time and records the exact seed sets and software versions.
- **SC-005**: The README, curriculum coverage, roadmap, reproduction matrix, and project alignment identify the lesson as a simulator analogue and preserve the separate paper-reproduction gates.
