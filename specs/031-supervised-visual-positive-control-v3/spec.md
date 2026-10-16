# Feature Specification: Supervised Visual Positive-Control Study v3

**Feature Branch**: `codex/supervised-visual-positive-control-v3`

**Created**: 2026-10-16

**Status**: Draft

**Input**: User request to review PR #6 and run a bounded, preregistered supervised visual positive-control study.

## User Scenarios & Testing

### User Story 1 - Explain the v2 visual-model failure (Priority: P1)

As a research reader, I need a verified audit of the existing supervised visual model so I can tell measured causes from possible causes.

**Why this priority**: The next experiment must address the observed failure without treating a failed benchmark as proof of a defective objective.

**Independent Test**: Compare the audit statements with the retained v2 code, split records, model records, action triplets, training history, and observation-control evidence.

**Acceptance Scenarios**:

1. **Given** the retained v2 records, **When** the audit is checked, **Then** every reported range and failure agrees with the raw records.
2. **Given** a cause that v2 did not isolate, **When** the report describes it, **Then** it is labeled as a hypothesis with the missing evidence stated.

### User Story 2 - Run a controlled visual positive-control study (Priority: P1)

As a research contributor, I need fixed baselines, seeds, thresholds, and stop rules so I can repeat the validation without changing the study after seeing its results.

**Why this priority**: A fresh, matched validation can distinguish visual-state recovery, action-conditioned prediction, and target parameterization.

**Independent Test**: Verify the protocol and implementation hashes were locked before training, then validate the raw run and action records against that lock.

**Acceptance Scenarios**:

1. **Given** three independent model/data seed pairs, **When** the validation stage runs, **Then** it evaluates the same five-level baseline hierarchy and the action-shuffling control on shared validation contexts.
2. **Given** a failed frozen gate, **When** the run completes, **Then** no independent test data, multi-step rollouts, or planning results are opened.

### User Story 3 - Verify and communicate the evidence (Priority: P2)

As a learner or portfolio reader, I need per-coordinate records and reproducible figures so I can judge the result without relying on a headline score.

**Why this priority**: A transparent negative or positive result is more useful than a single aggregate loss.

**Independent Test**: Regenerate each figure from retained CSV evidence and compare its byte hash with the committed figure.

**Acceptance Scenarios**:

1. **Given** a complete validation run, **When** its evidence is validated, **Then** all methods, seed pairs, starts, actions, and failures are present and finite.
2. **Given** unchanged evidence, **When** figures are regenerated twice, **Then** the two outputs match exactly and show the sample sizes and limits.

### Edge Cases

- If the exact transition or handcrafted visual control fails its frozen data-path check, stop before fitting learned models.
- If a fit is incomplete, non-finite, or exceeds the compute or storage limit, retain its status and do not replace its seed.
- If no learned visual dynamics family passes validation, do not generate independent test data or rollouts.
- If the action-shuffling control passes the action-fidelity gate, keep independent evaluation closed because the negative control did not validate action-specific learning.
- If a candidate predicts a state outside the simulator limits, retain the raw prediction and count it as clipping; do not hide it by clipping before scoring.

## Requirements

### Functional Requirements

- **FR-001**: The audit MUST distinguish observations verified from PR #6 code and raw records from explanations that remain hypotheses.
- **FR-002**: A versioned protocol MUST freeze the data split, independent seeds, method definitions, thresholds, selection rule, compute limit, failure rules, and evidence files before model training.
- **FR-003**: Validation MUST compare exact simulator dynamics from the handcrafted visual state, supervised regression on the same handcrafted features, a learned visual current-state estimator, the existing supervised visual dynamics model, and a residual-dynamics alternative.
- **FR-004**: Validation MUST include a training-only action-shuffling negative control and a single target-standardization ablation that changes no other training setting.
- **FR-005**: The study MUST report position and velocity separately, including prediction errors, action-contrast error, action ordering, clipping, action sensitivity, training curves, and gradient magnitudes. A candidate visual dynamics fit MUST also pass the frozen embedding-spread, effective-rank, and linear-probe checks.
- **FR-006**: Independent test data and multi-step rollouts MUST remain closed until the frozen validation gate passes. Planning MUST remain outside this feature.
- **FR-007**: Notebook 22, Notebook 23 v1, and v2 source, records, and figures MUST remain unchanged.
- **FR-008**: Retained validation evidence MUST include per-run and per-start values, failed or incomplete runs, provenance, and enough data to regenerate every figure.
- **FR-009**: Tests MUST check data-split isolation, fixed seeds, model action sensitivity, numeric failures, gate behavior, evidence completeness, and deterministic figure generation.

### Key Entities

- **Seed Pair**: One independent data seed matched to one model seed and reused across all candidate methods.
- **Validation Context**: One shared rendered two-frame input and simulator state used for action counterfactuals.
- **Model Run**: One method and seed pair with its status, training history, device, elapsed time, and per-coordinate validation metrics.
- **Action Triplet**: Actual and predicted next states for actions -1, 0, and +1 from one fixed context.
- **Gate Decision**: The frozen per-seed and per-family validation result that controls access to independent tests.
- **Evidence Manifest**: Relative file names, sizes, and SHA-256 hashes for retained results and figures.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The exact dynamics control matches simulator next-state labels within 1e-6, and the handcrafted visual control meets its preregistered position and velocity limits for all three validation data groups.
- **SC-002**: The validation record contains all 15 preregistered neural fits and all three handcrafted-feature regression fits, or records each incomplete fit and its reason without substitution.
- **SC-003**: A learned visual dynamics family opens the independent test stage only when at least two of its three seed pairs pass every frozen representation, state, action-fidelity, finite-output, and clipping threshold and the action-shuffling control fails its action-fidelity gate on at least two pairs.
- **SC-004**: If the gate fails, no test-seed data or multi-step rollout record is generated.
- **SC-005**: Every figure regenerates byte-for-byte from retained records, and each chart states its model, coordinate, sample count, metric, and whether it uses validation or gated held-out data.
- **SC-006**: The training stage stays within 60 GPU-minutes, 2 GiB peak allocated GPU memory, and 5 GiB of transient output on E:.

## Assumptions

- The study uses the existing synthetic MountainCar renderer and training examples; it does not add data, papers, or model families outside the requested hierarchy.
- The test stage, if opened, evaluates only the selected supervised visual dynamics family and diagnostic controls. It does not run a planner.
- Full frames, checkpoints, and transient arrays stay on E:. Only compact records, source, and figures are committed.
- Thresholds are set from existing v2 positive-control evidence and are not changed after v3 training starts.
