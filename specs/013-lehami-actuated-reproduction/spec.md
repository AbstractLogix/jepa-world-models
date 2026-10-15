# Feature Specification: LeHami Actuated-Pendulum Velocity Reproduction

**Feature Branch**: `013-lehami-actuated-reproduction`  
**Created**: 2026-10-14  
**Status**: In Progress  
**Input**: User request to reproduce published research findings as part of the tutorial roadmap.

## User Scenarios & Testing

### User Story 1 - Compare a paper result with a local run (Priority: P1)

A learner or contributor wants to check the published actuated-pendulum velocity result with the authors' data recipe, model settings, and evaluation method.

**Why this priority**: This is the smallest available experiment that directly tests a current paper claim with first-party code and a reproducible generated dataset.

**Independent Test**: Evaluate both named models on the held-out episode split, using three training seeds and the anchor 25-neighbor probe. The resulting report contains every seed result and compares the local mean and standard deviation with the values in Table 1.

**Acceptance Scenarios**:

1. **Given** the paper's actuated-pendulum data and split, **When** the two named models are trained with the paper settings and evaluated, **Then** the report records test-set angular-velocity MAE for each model, probe, and seed.
2. **Given** a training or evaluation run is missing or the data hash does not match the shipped manifest, **When** results are summarized, **Then** the report labels the attempt partial and lists the missing or changed condition.

### User Story 2 - Understand the strength and limits of the evidence (Priority: P2)

A learner wants to know what the result supports, what differs from the paper, and what the structural guarantee does not establish.

**Why this priority**: Clear limits prevent a small reproduction from being read as validation of the full method or of physical-system accuracy.

**Independent Test**: Review the report, roadmap, and coverage map. Confirm that paper values and local values are separate, the scope names one table result, and limitations remain explicit.

**Acceptance Scenarios**:

1. **Given** the local results, **When** a learner reads the report, **Then** it identifies the task, split, seeds, probe, compute conditions, deviations, and claim limits.
2. **Given** the paper's symplecticity claim, **When** the result is explained, **Then** the lesson states that volume preservation does not bound rollout error or establish physical validity.

### Edge Cases

- The regenerated data do not match the SHA-256 stored in the official split manifest.
- CUDA is unavailable, a run stops early, or fewer than three seeds finish for either model.
- A probe result is missing, non-finite, or far from the paper value; preserve the raw result and do not replace it with a frozen figure value.
- The local values differ from the paper while the official protocol and artifacts match; report the difference without changing settings after inspecting the test outcomes.

## Requirements

### Functional Requirements

- **FR-001**: The reproduction MUST target the actuated-pendulum anchor angular-velocity MAE measured by the 25-neighbor probe in Table 1 of the LeHami paper.
- **FR-002**: The comparison MUST include LeHami model `a0q` and the LeWM baseline `b0s`, with three training seeds, the authors' episode-level test split, and the published training settings.
- **FR-003**: The result record MUST keep per-seed values, aggregate mean and standard deviation, paper-reported values, run conditions, and deviations distinct.
- **FR-004**: The report MUST identify the paper, first-party code revision, model configurations, generated data hash, split manifest, probe, metric, and local compute conditions.
- **FR-005**: The reproduction MUST be labelled partial if any required seed, exact data hash, model setting, or evaluation output is missing or changed.
- **FR-006**: The roadmap and coverage map MUST identify this as a selected-result reproduction, not a full reproduction of the paper or a physical-robot result.
- **FR-007**: The existing LeWAM Push-T archive transfer and 450-episode evaluation MUST remain pending until separately selected.

### Key Entities

- **Paper claim**: The Table 1 test-split angular-velocity MAE for a named model and probe.
- **Run**: One model, training seed, official configuration, checkpoint, evaluation output, and compute record.
- **Dataset split**: The generated actuated-pendulum episodes and the authors' train, validation, and test episode lists.
- **Reproduction report**: The paper claim, local run values, deviations, evidence links, and status.

## Success Criteria

- **SC-001**: All six required model runs (two models across three seeds) have recorded evaluation outputs, or each missing run is listed and the attempt is labelled partial.
- **SC-002**: The report shows the paper's mean and standard deviation beside the local per-seed values, mean, and standard deviation for the same test metric.
- **SC-003**: A reviewer can identify the code revision, configurations, data hash, split, probe, hardware, software versions, runtime, and all deviations from the report.
- **SC-004**: Roadmap, reproduction matrix, curriculum coverage, and project task status agree about the scope and completion state.
