# Feature Specification: Released LeWAM Push-T Checkpoint Reproduction

**Feature Branch**: `010-lewam-pusht-reproduction`  
**Created**: 2026-10-14  
**Status**: Scoped; local evaluation has not started  
**Input**: Reproduce a published JEPA world-model finding with the official model, data, and evaluation protocol.

## User Scenarios & Testing

### User Story 1 - Re-evaluate the released Push-T checkpoint (Priority: P1)

A contributor wants to check whether the official LeWAM Push-T checkpoint gives the reported results when evaluated with the paper's settings.

**Why this priority**: The official project provides code, a checkpoint, a dataset, and a reproduction guide. This is a bounded way to compare a local result with a published finding without repeating large-scale training.

**Independent Test**: Run the three published evaluation modes for seeds 42, 0, and 1, with 50 episodes per mode and seed. Preserve every episode result and the exact run settings.

**Acceptance Scenarios**:

1. **Given** the pinned official code, checkpoint, and Push-T dataset, **When** the reactive policy, best-of-32 planner, and gradient planner run, **Then** each mode completes 50 episodes for seeds 42, 0, and 1.
2. **Given** the three-seed runs, **When** a contributor reads the result report, **Then** it shows the paper value beside each locally measured per-seed rate and the mean and standard deviation across seeds.
3. **Given** a local mean for an evaluation mode, **When** a contributor compares it with the paper value, **Then** the report labels it within or outside the paper's reported one-standard-deviation range. This label is not a statistical equivalence test.
4. **Given** the full protocol cannot run, **When** a contributor reads the report, **Then** it names the blocking error, completed partial work, and every deviation without calling the result reproduced.

### User Story 2 - Check artifact and protocol provenance (Priority: P2)

A reader wants to know whether the local evaluation used the official artifacts and what the measured result can support.

**Why this priority**: An evaluation result is not useful if the checkpoint, dataset, or evaluation protocol can change without being recorded.

**Independent Test**: Compare the report's source revisions and hashes with the official repositories and confirm the run used the same released checkpoint and evaluation settings.

**Acceptance Scenarios**:

1. **Given** the local run report, **When** a reader checks its sources, **Then** it lists the official code commit, checkpoint and dataset revisions, file hashes, licenses, and local software and hardware.
2. **Given** the evaluator samples episodes from the training dataset, **When** the result is described, **Then** the report states that the result is a checkpoint re-evaluation and does not claim held-out generalization or model training reproduction.

### Edge Cases

- The official checkpoint may fail to load under the pinned code or runtime. Record the exact error and stop before reporting scores.
- A missing or partial dataset archive must not produce a completed result.
- The evaluator is stochastic. Preserve all episode results and use all three stated seeds.
- The official guide says evaluation episodes are sampled from the training data. Do not describe them as held-out episodes.
- A CPU or GPU run may differ from the paper's hardware. Record the device and avoid claims of bitwise equality.

## Requirements

### Functional Requirements

- **FR-001**: The attempt MUST pin the official code repository to commit `60732e25ba172b117d0d0a476d3a9dcd5d30c03b`, the Push-T model repository revision `d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2`, and dataset repository revision `63346c94f4dbe62b4744261c6c99d63cbff3f6e6`.
- **FR-002**: The attempt MUST verify the checkpoint SHA-256 `e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb` and dataset archive SHA-256 `a075dcd49120844a1720cd3f650519f934a632b5c3e9a5a4e413ba8b020aaa43` before the full evaluation.
- **FR-003**: The evaluation MUST use the official Push-T settings: goal offset 25 steps, 50-action budget, 50 episodes per seed, and seeds 42, 0, and 1.
- **FR-004**: The evaluation MUST cover the reactive policy, best-of-32 planner, and gradient planner with the official settings: `lewam_policy`; `lewam_plan` with `best_of_k`; and `lewam_plan` with `grad`, `grad_all_k=true`, and `grad_tr=0.01`.
- **FR-005**: The report MUST retain all episode results, the per-seed success rates, the mean and standard deviation across three seeds, software versions, device, duration, source revisions, hashes, and deviations.
- **FR-006**: The report MUST compare local values with the paper's reported Push-T values: 85.3% ± 5.0 reactive, 95.3% ± 2.3 best-of-32, and 98.7% ± 1.2 gradient planning.
- **FR-007**: The report MUST state that the official evaluation samples episodes from training data. It MUST NOT claim held-out generalization or reproduction of model training.
- **FR-008**: If any required artifact or run is missing, the attempt MUST remain partial or blocked and MUST NOT be labeled a completed reproduction.
- **FR-009**: Downloaded data, checkpoints, logs, videos, and executed notebooks MUST remain outside version control.
- **FR-010**: The attempt MUST add no dependencies to the tutorial project's environment files.
- **FR-011**: The report MUST label each local mean as within or outside the paper's reported one-standard-deviation range. It MUST state that this is a comparison rule, not a significance or equivalence test.

### Key Entities

- **Paper claim**: A named LeWAM Push-T success-rate result, evaluation mode, and reported summary.
- **Artifact pin**: An official code revision, checkpoint or dataset revision, content hash, and license.
- **Evaluation run**: One evaluation mode and seed, with its 50 episode outcomes, environment, device, and duration.
- **Reproduction report**: A dated record of paper values, local measurements, artifact provenance, failures, and claim limits.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The report records verified code, model, and dataset revisions and both required content hashes before any complete evaluation is reported.
- **SC-002**: The full attempt contains 450 episode results: three evaluation modes × three seeds × 50 episodes. Any missing run is stated and the attempt remains partial.
- **SC-003**: The report gives three per-seed success rates and one mean and standard deviation across seeds for each evaluation mode.
- **SC-004**: Each local summary appears next to the paper-reported value and is labeled as a released-checkpoint evaluation on sampled training episodes.
- **SC-005**: Data, checkpoints, logs, videos, and temporary source checkouts remain outside the Git worktree.
- **SC-006**: Each of the three local mode means has an explicit within/outside comparison to the paper's reported one-standard-deviation range.

## Assumptions

- The official evaluation guide and released checkpoint are the target. This attempt does not retrain LeWAM.
- The official Push-T archive is 13.8 GB compressed and expands to about 43.1 GiB. Enough storage and transfer time are needed.
- The public model and dataset cards declare MIT licenses. The attempt will follow the official repository's dependency and data instructions.
- The local host has an NVIDIA RTX 5070 with 12 GB VRAM, while the current Python environment has CPU-only PyTorch. The supported local evaluation device is not confirmed yet.
- The evaluation episodes are sampled from the official training dataset, as stated in the official reproduction guide.

## Scope

### In Scope

- Pin the official code, Push-T checkpoint, and Push-T dataset.
- Run the official three-mode, three-seed Push-T checkpoint evaluation if the local runtime and data transfer permit it.
- Record raw episode outcomes and the comparison with paper values.

### Out of Scope

- Retraining LeWAM or reproducing its pretraining and training data pipeline.
- Reproducing robot experiments or the results for other environments.
- Claiming held-out generalization from episodes sampled from the training dataset.
- Adding external dependencies to the tutorial project's runtime.
