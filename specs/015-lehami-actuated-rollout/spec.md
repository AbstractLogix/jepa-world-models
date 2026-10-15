# Feature Specification: LeHami Seed-0 Actuated Rollout Evaluation

**Feature Branch**: `015-lehami-actuated-rollout`  
**Created**: 2026-10-15  
**Status**: Implemented locally; one seed per model; no paper reproduction claimed  
**Input**: Evaluate the already completed seed-0 LeHami and LeWM checkpoints on the official held-out actuated-pendulum rollout with the authors' K=1-fitted 25-neighbor probe. Use the pinned code, data, and split. Add no training seeds and do not claim a three-seed reproduction.

## User Scenarios & Testing

### User Story 1 - Measure prediction across a rollout (Priority: P1)

A learner wants to compare how the LeHami and LeWM seed-0 checkpoints decode angular velocity as their latent predictions advance through an actuated-pendulum episode.

**Why this priority**: The existing local result measures only a K=0 anchor probe. A frozen K=1 probe over multiple rollout steps gives direct local evidence about prediction accuracy alongside the lesson's structural example.

**Independent Test**: Run the authors' standard evaluator for both saved checkpoints at horizon 76. The report includes held-out angular-velocity MAE for selected rollout steps, the per-step raw outputs, run conditions, and the explicit one-seed limit.

**Acceptance Scenarios**:

1. **Given** the completed seed-0 checkpoints and the hash-matched actuated-pendulum data, **When** the official evaluator runs, **Then** it fits the 25-neighbor probe at K=1 on the training split, applies it frozen across the rollout, and records held-out metrics through K=76.
2. **Given** both model evaluations, **When** the local values are reported, **Then** the report labels them as one seed per model and keeps the paper's three-seed mean and standard deviation separate.
3. **Given** a completed local rollout evaluation, **When** Notebook 15 is updated, **Then** it distinguishes structural volume preservation, measured rollout prediction, and unmeasured planning or physical validity.

### Edge Cases

- The generated dataset hash differs from the shipped manifest. Stop before fitting probes or scoring.
- A checkpoint or evaluator output is missing or non-finite. Preserve the available files and label the result incomplete.
- The predicted probe values differ from the paper curve. Report the result without changing configs, probes, or horizons after seeing test outcomes.
- The official evaluator also writes train-split output. Use the held-out test rows for the paper comparison and retain both split labels in the raw files.

## Requirements

### Functional Requirements

- **FR-001**: The evaluation MUST use the existing full seed-0 checkpoints for LeHami `a0q` and LeWM `b0s`, from the pinned official LeHami code checkout.
- **FR-002**: The evaluation MUST verify the dataset hash against the official manifest and use the authors' episode split and recorded action sequence.
- **FR-003**: The evaluation MUST use the authors' standard rollout mode, horizon 76, a 25-neighbor probe fitted at K=1 on training data, and the same frozen probe at each rollout step.
- **FR-004**: The report MUST provide test-split angular-velocity MAE by rollout step, with values for both models and selected horizons, and preserve the full raw per-step outputs.
- **FR-005**: The report MUST identify the source revision, checkpoint hashes, data hash, probe, split, seed, runtime, hardware, command, wall time, and any deviations.
- **FR-006**: The paper's three-seed mean and standard deviation MUST remain separate from the single local seed-0 measurements. Do not call the finding reproduced.
- **FR-007**: The report and Notebook 15 MUST state that rollout prediction metrics do not establish useful planning, control performance, or physical-system validity.
- **FR-008**: The project roadmap and coverage records MUST identify the new local rollout evidence and keep seeds 1 and 2 open for both models.

### Key Entities

- **Rollout checkpoint**: One seed-0 model checkpoint (`a0q` or `b0s`) evaluated with the matching resolved model configuration.
- **Frozen probe**: A 25-neighbor model fitted to K=1 predictions from training episodes and reused without refitting at later rollout steps.
- **Per-step metric**: Held-out test-split angular-velocity MAE for one model, probe, rollout step, and seed.
- **Evidence scope**: The paper's three-seed statistics and the local seed-0 measurements, kept separate.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Both saved seed-0 checkpoints complete the official standard evaluation through K=76 without new training.
- **SC-002**: The saved raw results include held-out probe metrics at every evaluated step, and the selected-horizon summary contains K=1, 5, 10, 20, 30, 50, and 76 when those rows are present in the official output.
- **SC-003**: The report records all required hashes, source revision, seed, split, probe, runtime, hardware, wall time, command, and deviations.
- **SC-004**: The report and notebook label the local result partial and do not report a local three-seed mean or standard deviation.
- **SC-005**: The roadmap, curriculum coverage, and project task list link to the rollout evidence and leave the four additional training runs pending.

## Assumptions

- Existing seed-0 checkpoints and the hash-matched generated dataset remain available in their temporary homes.
- The official standard evaluator is the source of the probe and rollout calculations; no replacement evaluation code is needed.
- This feature extends seed-0 evaluation only. It does not authorize training seeds 1 and 2, planning evaluation, or the LeWAM Push-T archive transfer.
- The local rollout run can be done on the existing CUDA host without changing the repository dependency set.
