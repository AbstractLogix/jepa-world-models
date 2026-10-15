# Feature Specification: PiJEPA CAST Table 2 Reproduction

**Feature Branch**: `[017-pijepa-cast-reproduction]`  
**Created**: 2026-10-15  
**Status**: Scoped; execution is blocked at preflight  
**Input**: User goal T041: scope one exact policy-guided planning paper result and record source, access, compute, and stop conditions.

## User Scenarios & Testing

### User Story 1 - Check the policy-guided planning result (Priority: P1)

As a learner, I want to compare the published PiJEPA result with its policy and planning baselines so that I can inspect the measured effect of policy-guided planning.

**Why this priority**: This is the project's selected paper-reproduction candidate for the policy-prior lesson.

**Independent Test**: A reviewer can identify the exact paper table, primary metric, methods, data, evaluation cases, run count, and local status from the feature and result report.

**Acceptance Scenarios**:

1. **Given** licensed, pinned assets and a supported host, **When** all named methods are evaluated on the same fixed CAST validation cases, **Then** the report shows every local run beside the Table 2 values and reports the primary V-JEPA-2 ATE XY RMSE metric in meters.
2. **Given** a required data term, trained checkpoint, evaluation-case list, or compute resource is unavailable, **When** preflight ends, **Then** no large transfer or training run starts and the report lists the blocker and labels the result `Not run` or `partial`.
3. **Given** the paper or code gives no training seed count, **When** a local comparison is planned, **Then** it uses the project default of at least five independent training seeds for stochastic training or labels any smaller run exploratory or partial.

---

### User Story 2 - Understand the limits of the reproduction (Priority: P2)

As a learner, I want paper values, local measurements, and unavailable inputs shown separately so that I do not mistake an analogue or incomplete run for a reproduced result.

**Why this priority**: The paper, code, data, model weights, and hardware do not have the same access status.

**Independent Test**: A reviewer can tell which parts were run, which remain unavailable, and why the local evidence does or does not reproduce the selected finding.

**Acceptance Scenarios**:

1. **Given** a missing license, model asset, data split, or compute requirement, **When** the report is read, **Then** it names the missing input and does not infer permission or substitute an unverified asset.
2. **Given** local results exist, **When** they are summarized, **Then** the report labels paper-reported and locally measured values separately and links raw per-run records.

### Edge Cases

- The CAST files are downloadable but the dataset card does not state a license or usage terms.
- The official code expects user-supplied policy and world-model checkpoint paths, but no PiJEPA-trained checkpoints are linked in the reviewed repository.
- The paper reports results on validation episodes, while the README example requests 1,000 examples and the paper does not identify the exact evaluation-case list.
- The local host has one RTX 5070 with 12 GB of memory, while the paper reports four H200 GPUs.
- A run completes for fewer seeds or fewer evaluation cases than planned, or a method fails during training or evaluation.

## Requirements

### Functional Requirements

- **FR-001**: The reproduction MUST target the V-JEPA-2 ViT-L result in Table 2 of *Policy-Guided World Model Planning for Language-Conditioned Visual Navigation*, arXiv:2603.25981v1.
- **FR-002**: The primary metric MUST be ATE XY RMSE in meters. The local report MUST also preserve the paper's other Table 2 metrics when the official evaluator returns them.
- **FR-003**: The comparison MUST include PiJEPA, Octo-WM, Octo, and default MPPI with the same encoder, task, validation cases, and stated compute budget.
- **FR-004**: The report MUST show the paper's Table 2 ATE XY RMSE values separately: PiJEPA 1.65 m, Octo-WM 1.67 m, Octo 1.72 m, and default MPPI 1.87 m.
- **FR-005**: Before any dataset or checkpoint transfer, the feature MUST record the exact source revision, content hash, license or usage terms, asset size, evaluation split, and access date for every required input.
- **FR-006**: Training or evaluation MUST NOT start until dataset terms, required model assets, the fixed evaluation cases, and a sufficient compute and storage plan are verified.
- **FR-007**: The local comparison MUST use at least five independent training seeds when training is stochastic unless the paper specifies another run count. A smaller run MUST be labeled exploratory or partial and MUST list missing runs.
- **FR-008**: The report MUST preserve raw per-run and per-case values, run failures, software and hardware, elapsed time, and deviations from the paper.
- **FR-009**: The report MUST NOT describe paper values, code inspection, setup checks, or an incomplete run as a local reproduction.
- **FR-010**: The README, roadmap, reproduction matrix, coverage map, project alignment, and remaining-work records MUST agree on the target and evidence status after the attempt.

### Key Entities

- **Paper result**: One named Table 2 metric, method, encoder, and published value.
- **Asset record**: A source revision, file hash, size, license or usage terms, and access date for code, data, or model weights.
- **Evaluation case**: One fixed CAST validation example and its shared input to each compared method.
- **Training run**: One method and encoder trained with one independent seed, source configuration, checkpoint, and compute record.
- **Evaluation run**: One method and checkpoint evaluated with a pinned evaluator, fixed case manifest, and recorded compute conditions; it can use a frozen checkpoint without a training-run record.
- **Reproduction report**: The paper target, local values, raw records, deviations, missing inputs, and claim status.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The selected Table 2 row, primary metric, methods, and paper values are pinned to a primary source before any run.
- **SC-002**: Every transferred dataset, model, and source artifact has a recorded revision or identifier, hash where available, license or usage terms, and size.
- **SC-003**: If an access or resource gate fails, the repository records the exact blocker and no large dataset transfer or training run is started.
- **SC-004**: A completed comparison contains all planned independent runs and shared evaluation cases, raw per-run values, variation, and compute conditions; any missing run keeps the claim partial.
- **SC-005**: The final report and project status documents distinguish paper-reported values from local measurements and identify whether the selected result was reproduced.

## Assumptions

- The exact target is the V-JEPA-2 encoder comparison in Table 2, not all four cross-encoder combinations.
- The PiJEPA repository's `main` revision and the CAST dataset's `main` revision are pinned in the plan as review snapshots; a future run must recheck them before use.
- The current project evaluation protocol applies. It requires at least five independent training seeds for a stochastic comparison when the paper does not state a run count.
- A public download link does not establish permission to use the CAST data. The missing data terms are a stop condition.
- This feature does not require a physical robot. It targets the paper's offline CAST validation evaluation.
