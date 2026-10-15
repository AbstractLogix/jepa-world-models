# Feature Specification: Bounded LpWM PushT Reproduction

**Feature Branch**: `003-lpwm-pusht-reproduction`

**Created**: 2026-10-14

**Status**: Attempt recorded; training did not complete, so no paper finding was reproduced

**Input**: User asked the repository to try to reproduce research findings and validate them.

## User Scenarios & Testing

### User Story 1 - Run a paired sparse and dense model comparison (Priority: P1)

A learner wants to test one precise claim from the LpWM paper. They need the same PushT data, predictor family, latent dimension, training conditions, and planning metric for sparse LpWM and dense LeWM.

**Why this priority**: This is the smallest author-configured experiment that can test whether the reported sparse-model advantage appears at a named intermediate predictor capacity.

**Independent Test**: Train one sparse and one dense MLP∘LTV model at latent dimension 768 from the pinned author configuration. Confirm that both use the verified PushT data file, complete the stated training schedule, and save model and configuration records.

**Acceptance Scenarios**:

1. **Given** the source revision, data file, and terms status are recorded, **When** a learner runs the paired experiment, **Then** the report identifies both model settings, the training seed, and any deviation from the paper.
2. **Given** either training run fails, **When** the run is recorded, **Then** its failure and available logs are reported without presenting the pair as complete.

---

### User Story 2 - Measure planning success across evaluation seeds (Priority: P2)

A learner wants to compare the trained models using the paper's planning outcome. They need the individual evaluation results and a summary that does not hide variability.

**Why this priority**: Training alone does not test the paper's planning claim. The planned metric is the percentage of PushT evaluation episodes that reach the goal.

**Independent Test**: Evaluate each completed model on 50 trajectories for each of three recorded planner seeds. Report each success rate, the mean, and the standard deviation for each model.

**Acceptance Scenarios**:

1. **Given** both trained checkpoints are available, **When** each planner run completes, **Then** the report records the seed, number of episodes, planning settings, and success rate.
2. **Given** one or more planner runs fail or have fewer than 50 episodes, **When** results are summarized, **Then** missing runs remain visible and the result is labeled incomplete.

---

### User Story 3 - Interpret the result within its evidence boundary (Priority: P3)

A learner wants to know whether the local result supports the paper's finding. They need the exact claim, the local result, the limits of the attempt, and the data-access record in one place.

**Why this priority**: A single predictor and dimension can test one reported setting, but it cannot establish the paper's full range or its maximum reported advantage.

**Independent Test**: Read the reproduction report and verify that paper values, local measurements, seed choices, data terms, and deviations are separate and complete.

**Acceptance Scenarios**:

1. **Given** all required runs are complete, **When** a learner reads the summary, **Then** the attempt is labeled a partial reproduction of one configuration, not a full reproduction of the paper.
2. **Given** the OSF data record has no license metadata, **When** access and use are described, **Then** the report states that uncertainty and confirms the dataset was not added to the repository.

## Edge Cases

- The OSF dataset file is unavailable, changes size, or fails its published checksum.
- The OSF record does not state a data license or other use terms.
- The upstream training or planning command fails on the local CPU-only machine.
- One model trains while the other fails, leaving no complete paired comparison.
- A planner run completes fewer than 50 episodes or produces no success metric.
- The local result differs from the plotted paper result or does not show a sparse-model advantage.
- The upstream code or reproduction configuration changes after the reviewed commit.

## Requirements

### Functional Requirements

- **FR-001**: The attempt MUST name one paper claim and one predictor-capacity setting before training. It MUST identify the attempt as partial if it covers only one setting.
- **FR-002**: The attempt MUST pin the paper version, author-code revision, configuration, dataset source, file size, checksum, and available terms before interpreting results.
- **FR-003**: The attempt MUST compare the authors' sparse LpWM and dense LeWM settings for MLP∘LTV at latent dimension 768 using the same PushT data and planning protocol.
- **FR-004**: The attempt MUST record the training seed and every training setting that can affect the comparison, including method-specific hyperparameters.
- **FR-005**: The attempt MUST report closed-loop planning success for 50 trajectories at each of three planner seeds, with per-seed values, mean, and standard deviation. Any incomplete run MUST remain visible.
- **FR-006**: The report MUST separate paper-reported values from local measurements and state whether the local result agrees with the selected paper claim.
- **FR-007**: Dataset files, model checkpoints, and raw generated outputs MUST stay outside version control. The report MUST NOT claim redistribution rights when terms are unknown.
- **FR-008**: The report MUST state hardware, software versions, elapsed time, memory use when available, data access limits, and all deviations from the paper and author script.

### Key Entities

- **Experiment definition**: Paper claim, paper version, author-code revision, model arm, predictor family, latent dimension, hyperparameters, seeds, and planning protocol.
- **Dataset record**: OSF source, archive name, file size, published checksum, local verification status, and terms status.
- **Training run**: Model arm, configuration, seed, outcome, elapsed time, resource use, and checkpoint location.
- **Planning run**: Model checkpoint, planner seed, episode count, planning settings, success count, and success rate.
- **Reproduction report**: Paper result, local run table, deviations, incomplete runs, and evidence-boundary label.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The source review identifies one exact author configuration for each arm and records the upstream code revision before any training run.
- **SC-002**: The local PushT archive matches the published size and SHA-256, or the attempt stops and records the mismatch.
- **SC-003**: Both model arms complete their configured training runs, or each failure is recorded with its available evidence.
- **SC-004**: Each reported model summary contains three planner runs of 50 episodes, unless the report clearly marks the attempt incomplete.
- **SC-005**: The final report labels this single-setting attempt as partial and makes no claim that it reproduces the paper's 36%–45% MLP∘LTV range.

## Assumptions

- The selected paper claim is the reported PushT planning-success advantage of sparse LpWM over dense LeWM for MLP∘LTV predictors. The paper reports a 36%–45% success-rate advantage across dimensions; this attempt checks one dimension only.
- The chosen setting is D=768. The official reproduction script provides separate selected hyperparameters for sparse and dense arms.
- The paper reports one training seed and three planning seeds, but does not name the seed values in its prose. Use the author configuration's training default and record local planner seeds 99, 100, and 101 as explicit local choices.
- The OSF node is public and lists the PushT archive, but the OSF API has no license metadata. Keep the data local, do not redistribute it, and report the unknown terms.
- Before examining local scores, use this directional decision rule: the selected claim is supported at this one setting only if the sparse arm has a higher mean success rate than the dense arm over the three local planner seeds. Report the three paired seed values and the difference. This rule does not test the paper's full 36%–45% range or establish statistical significance.
- No code, data, checkpoints, or plots from the upstream repository are copied into the tutorial repository.
