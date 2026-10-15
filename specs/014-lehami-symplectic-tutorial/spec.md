# Feature Specification: LeHami Symplectic Dynamics Tutorial

**Feature Branch**: `014-lehami-symplectic-tutorial`  
**Created**: 2026-10-15  
**Status**: In Progress  
**Input**: Add a tutorial that explains the LeHami and LeWM comparison, shows what symplectic structure guarantees, and separates that guarantee from prediction accuracy and control evidence. Use the completed seed-0 local results. Keep the remaining four runs pending.

## User Scenarios & Testing

### User Story 1 - See what a symplectic update preserves (Priority: P1)

A learner wants to understand why a structured latent update can preserve phase-space volume, and how that property differs from prediction quality.

**Why this priority**: The structural guarantee is the central idea of the paper and can be demonstrated with a small, local calculation that does not need research data or a trained model.

**Independent Test**: Run the tutorial from top to bottom on CPU. The numerical example reports the symplectic map's determinant and transformed polygon area, and contrasts these results with an explicitly labeled Euler example.

**Acceptance Scenarios**:

1. **Given** the tutorial's one-dimensional Hamiltonian example, **When** a learner applies the leapfrog update, **Then** the Jacobian determinant and polygon area are preserved within the documented numeric tolerance.
2. **Given** the Euler comparison, **When** a learner compares the same quantities, **Then** the tutorial reports its measured result and does not present Euler as the LeWM model.

### User Story 2 - Interpret the paper and local evidence (Priority: P2)

A learner wants to compare the paper's actuated-pendulum result with the current local run and know what is still unverified.

**Why this priority**: This prevents a one-seed probe or a structural theorem from being presented as evidence of a reproduced paper benchmark or useful control.

**Independent Test**: Review the paper/local comparison table and claim-limit section. The table separates the three-seed paper statistics from the one-seed-per-model local measurements and names the measured K=0 probe.

**Acceptance Scenarios**:

1. **Given** the published Table 1 values and local seed-0 measurements, **When** a learner reads the comparison, **Then** each value is labeled by source, seed count, metric, and task.
2. **Given** that seeds 1 and 2 are not complete for either model, **When** the lesson states its status, **Then** it calls the evidence partial and does not claim agreement, disagreement, rollout quality, planning value, or physical validity.

### Edge Cases

- A notebook run has no access to the temporary research checkout or training outputs. The CPU demonstration must still complete; measured values must remain visibly recorded evidence rather than be presented as values recomputed by the demo.
- Numeric rounding can make an exactly area-preserving map display a small nonzero error. The notebook must report its tolerance and avoid treating floating-point roundoff as a violation.
- The actuated Hamiltonian changes with the applied action. The lesson must not imply that symplecticity means the learned Hamiltonian is conserved across a changing action sequence.

## Requirements

### Functional Requirements

- **FR-001**: The tutorial MUST define phase-space, symplectic structure, and volume preservation in learner-facing terms before using them in the analysis.
- **FR-002**: The tutorial MUST show a deterministic numerical example in which a leapfrog update preserves the Jacobian determinant and the area of a closed polygon within a stated tolerance.
- **FR-003**: The tutorial MUST separate the structural property of the update from measured prediction error, control performance, and physical validity.
- **FR-004**: The tutorial MUST identify the LeHami paper, its official source repository, and the actuated-pendulum K=0 25-neighbor probe as the evidence scope.
- **FR-005**: The tutorial MUST present the paper's three-seed statistics separately from the local LeHami and LeWM seed-0 measurements.
- **FR-006**: The tutorial MUST label the local comparison partial while seeds 1 and 2 are missing for both models, and MUST NOT call the paper result reproduced.
- **FR-007**: The tutorial MUST explain that the selected anchor probe does not measure rollout quality, planning value, or physical-system accuracy.
- **FR-008**: The numerical demonstration MUST run locally on CPU without downloads, checkpoints, or robot hardware.
- **FR-009**: The tutorial and project coverage records MUST identify the lesson as a mathematical analogue with partial paper-linked evidence, not a full paper reproduction.

### Key Entities

- **Symplectic update**: A discrete map over position-like and momentum-like latent coordinates whose Jacobian preserves the canonical phase-space form and has determinant one.
- **Paper result**: The reported LeHami and LeWM held-out angular-velocity MAE statistics from the paper's actuated-pendulum K=0 25-neighbor probe.
- **Local result**: One measured seed-0 value for each model on the official data split and probe; not a local mean or estimate of variability.
- **Evidence scope**: The exact task, metric, run count, source, and limits supported by a result.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A learner can run every code cell on CPU without downloading data or model weights.
- **SC-002**: For each displayed leapfrog step size, the absolute Jacobian-determinant error and polygon-area error are at most `1e-12` in the documented float64 calculation.
- **SC-003**: The comparison shows both paper values (0.419 ± 0.045 rad/s for LeHami and 2.508 ± 0.002 rad/s for LeWM) and both local seed-0 values (0.456258 and 2.504686 rad/s), with their different run counts and status.
- **SC-004**: The notebook explicitly states that one seed per model cannot reproduce the paper's three-seed statistics and that the K=0 probe does not test rollout or control performance.
- **SC-005**: README, roadmap, coverage map, and project tasks identify the lesson and preserve the open four-run follow-up.

## Assumptions

- The seed-0 values and run conditions in the [LeHami reproduction report](../../docs/reproductions/lehami-actuated-velocity.md) remain the project record for this lesson.
- The tutorial uses a simple harmonic oscillator only to explain numerical structure. It is not a learned JEPA, a LeHami implementation, or a reproduction of the paper's experiment.
- The official LeHami paper and repository are the primary sources for the method and reported result.
- The four remaining training and probe runs remain pending and are out of this feature's scope.
