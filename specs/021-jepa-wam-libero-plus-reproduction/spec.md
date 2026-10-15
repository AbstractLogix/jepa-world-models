# Feature Specification: JEPA-WAM LIBERO-Plus Checkpoint Re-evaluation

**Feature Branch**: 021-jepa-wam-libero-plus-reproduction | **Created**: 2026-10-15 | **Status**: Preflight recorded; evaluation pending

**Input**: Project roadmap item for joint predictive and action-generating policies.

## User Scenarios & Testing

### User Story 1 - Re-check the released JEPA-WAM result (Priority: P2)

As a learner, I want a source-pinned evaluation of the released JEPA-WAM checkpoint on LIBERO-Plus so that I can compare a local result with the paper's reported robustness result.

**Independent Test**: Run the upstream one-task, one-action environment diagnostic in an isolated environment. If it passes, run the full spatial-suite evaluation with the paper's seven categories and one trial per task.

**Acceptance Scenarios**:

1. **Given** the released checkpoint, **When** the diagnostic starts one LIBERO-Plus task and produces one action, **Then** the report records the source revisions, artifact hashes, environment, renderer, GPU memory, and result as a pipeline smoke only.
2. **Given** a passing diagnostic and acceptable measured runtime, **When** the full spatial suite runs, **Then** it evaluates all 2,402 task variants, all seven categories, and one trial per task without task or episode-step caps.
3. **Given** complete evaluation output, **When** the result is compared with the paper, **Then** the report preserves all task outcomes, category rates, the aggregate score, compute, and deviations. It does not call a released-checkpoint evaluation a training reproduction.
4. **Given** a failed environment, artifact, runtime, or memory gate, **When** evaluation stops, **Then** the report records the failure and makes no paper-result claim.

## Requirements

### Functional Requirements

- **FR-001**: The target MUST be the JEPA-WAM Table 2 LIBERO-Plus result from [arXiv:2608.09381](https://arxiv.org/abs/2608.09381), using the author README's spatial-suite, all-category, one-trial command.
- **FR-002**: The evaluation MUST use the pinned JEPA-WAM source, checkpoint revision, LIBERO-Plus source, and required inference artifacts in the [preflight report](../../docs/reproductions/jepa-wam-libero-plus.md).
- **FR-003**: The one-task, one-action diagnostic MUST pass before the full 2,402-task spatial evaluation starts.
- **FR-004**: A full run MUST include every task variant and category, with one trial per task, and MUST preserve per-task outcomes and run metadata.
- **FR-005**: Reports MUST state the difference between the paper's 60,000-step, global-batch-128 training and the released recipe's 40,000-step, global-batch-256 checkpoint. The evaluation MUST be labeled as a checkpoint re-evaluation, not an independent training reproduction.
- **FR-006**: The evaluation MUST NOT download training RLDS data, other LIBERO-Plus suites, or the complete 373 GB checkpoint repository.
- **FR-007**: The project README, curriculum coverage, roadmap, reproduction matrix, project research, alignment, remaining work, and task ledger MUST separate this evaluation from the CPU tutorial analogues and from V-JEPA Policy, JEPA-WM, and LeWAM.

### Out of Scope

- Re-training JEPA-WAM with the paper's eight-GPU recipe.
- Evaluating the other three LIBERO-Plus suites.
- Running the π0.5 integration or any other JEPA-WAM-named paper.
- Adding the full benchmark assets, checkpoints, or simulator source to Git.
- Claiming a matched baseline comparison without the baseline checkpoint and protocol.

## Success Criteria

- **SC-001**: The upstream one-task, one-action diagnostic either passes or produces a recorded, actionable failure with no benchmark claim.
- **SC-002**: If the full run is completed, its saved outcomes cover all 2,402 spatial-suite task variants exactly once and include all seven perturbation categories.
- **SC-003**: The report states per-category rates, overall average, source and artifact identities, runtime, peak GPU memory, and deviations.
- **SC-004**: All project indexes identify this as a released-checkpoint re-evaluation and leave the 60,000-step training result unclaimed.

## Assumptions

- The author README's spatial-suite command is the intended operational path for its reported 79.2% score.
- The full run uses the benchmark's documented one-trial-per-task protocol.
- Training data is not needed to evaluate the released policy.
- A local result may differ because the released checkpoint and training recipe do not match the paper's stated training steps and batch size.
