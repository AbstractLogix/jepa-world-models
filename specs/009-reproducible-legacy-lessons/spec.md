# Feature Specification: Reproducible Legacy Lesson Evidence

**Feature Branch**: `009-reproducible-legacy-lessons`  
**Created**: 2026-10-14  
**Status**: Complete  
**Input**: User goal to give every lesson reproducible local evidence and keep the curriculum aligned with the shared evaluation protocol.

## User Scenarios & Testing

### User Story 1 - Run the original lessons locally (Priority: P1)

A learner can run notebooks 01 and 04–08 from a clean kernel on a supported local CPU environment. The notebook states what the run demonstrates and records enough settings to repeat it.

**Why this priority**: These lessons are part of the current curriculum, but the coverage review has no recent execution evidence for them.

**Independent Test**: Execute each notebook from the first cell through the last cell in the documented environment. Confirm each run completes or records a specific failure and limit.

**Acceptance Scenarios**:

1. **Given** the project dependencies and required public data or weights, **When** a learner runs notebook 01 or 04 from a clean kernel, **Then** every cell completes and the notebook records its seed, data/model source, device, and result boundary.
2. **Given** a CPU-only runtime, **When** a learner runs notebooks 05–08, **Then** each notebook completes with fixed environment, data, and model seeds and reports its measured outputs.
3. **Given** a notebook uses synthetic targets, training examples, or one training seed, **When** a learner reads its result, **Then** the notebook labels those limits and does not present them as a paper result or a stable method comparison.

### User Story 2 - Compare planning methods on matched cases (Priority: P1)

A learner can compare the planning methods in notebooks 05 and 06 on the same fixed CartPole initial states. The learner can inspect every episode result and the run conditions.

**Why this priority**: These notebooks compare policies. Different initial states or unseeded random actions can hide whether the method caused a difference.

**Independent Test**: Run each notebook twice from a clean kernel. Confirm each compared method uses the same declared reset seeds, each evaluation has at least 20 episodes, and every episode result appears in the run output.

**Acceptance Scenarios**:

1. **Given** the random policy and planned policy in notebook 05, **When** both policies run, **Then** they use the same 25 reset seeds and report each episode return.
2. **Given** random shooting and gradient planning in notebook 06, **When** both planners run, **Then** they use the same 20 reset seeds and report each episode return.
3. **Given** a planner uses random candidates, **When** the notebook runs again with the same seed, **Then** it uses the same candidate sequence for each episode.

### User Story 3 - Keep evidence and curriculum records aligned (Priority: P2)

A contributor can find one dated report for the local execution of notebooks 01 and 04–08 and can see the same status and limits in the curriculum coverage and project records.

**Why this priority**: Local measurements are useful only when learners can distinguish them from paper-reported results and find the run conditions.

**Independent Test**: Compare the report with the executed notebook outputs, `docs/CURRICULUM_COVERAGE.md`, and the project alignment and remaining-work records.

**Acceptance Scenarios**:

1. **Given** all notebook runs, **When** a contributor reads the report, **Then** it records software versions, CPU/GPU, runtime, seeds, raw measurements, errors, and known limits for each notebook.
2. **Given** these local runs, **When** a contributor reads curriculum or research documents, **Then** every local result is labeled as illustrative or exploratory and remains separate from paper benchmarks.

### Edge Cases

- A missing model weight or data archive must produce a clear failure record. The notebook must not report a completed result.
- A policy run may end early or reach the environment time limit. Record the return for that episode and keep it in the summary.
- A fixed seed may still produce platform-specific numeric differences. Record the actual software and device versions; do not claim bitwise portability.
- An ablation grid with one training seed is exploratory. Episode variation must not be described as training-seed uncertainty.

## Requirements

### Functional Requirements

- **FR-001**: Notebook 01 MUST use fixed random seeds for its generated views and model initialization. Its covariance ellipse MUST work with real symmetric covariance matrices on the supported NumPy versions.
- **FR-002**: Notebook 04 MUST use a fixed data subset and model seed, pin and record the ImageNet-1K `timm/resnet18.a1_in1k` feature source and license, and evaluate its synthetic brightness-and-edge descriptor task on held-out images. It MUST label the task as an analogue, not text retrieval or a VL-JEPA result.
- **FR-003**: Notebook 05 MUST seed offline trajectory collection and MUST evaluate random and planned policies on the same 25 fixed initial-state seeds.
- **FR-004**: Notebook 06 MUST evaluate random-shooting and gradient planners on the same 20 fixed initial-state seeds and MUST seed random planner candidates per episode.
- **FR-005**: Notebook 07 MUST seed action sampling and environment resets. Its one-step metric MUST state whether it measures training or held-out transitions.
- **FR-006**: Notebook 08 MUST seed its collected data, model initialization, and planner randomness. Every ablation setting MUST use the same fixed evaluation seeds, and the result MUST state that one training seed does not establish a reliable ranking.
- **FR-007**: The run report MUST record raw per-episode or per-setting results, settings, software versions, device, runtime, failures, and deviations for notebooks 01 and 04–08.
- **FR-008**: The curriculum coverage and project alignment records MUST match the current code and report. They MUST keep illustrative results separate from paper-reported findings.
- **FR-009**: Generated notebook outputs and checkpoints MUST remain outside version control.
- **FR-010**: The work MUST add no new project dependency or require robot hardware or private data.

### Key Entities

- **Run configuration**: The notebook, seed values, data subset, model source, environment, and compute used for one execution.
- **Evaluation episode**: A fixed initial-state seed, method, episode return, and termination state.
- **Ablation result**: One configuration, one training seed, fixed evaluation seeds, and per-episode outcomes.
- **Evidence report**: Dated local measurements, environment details, errors, and claim limits linked to the notebook sources.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Notebooks 01 and 04–08 each complete from a clean kernel on the supported CPU environment, or the report records the exact blocking error and the affected notebook.
- **SC-002**: Notebooks 05 and 06 evaluate both methods on identical reset-seed lists with at least 20 episodes and preserve every episode return.
- **SC-003**: Notebook 08 uses the same fixed evaluation seeds for every ablation setting and identifies its single training seed in the result table.
- **SC-004**: The report records software versions, device, runtime, seeds, raw results, and failures for all six notebooks.
- **SC-005**: The curriculum coverage and project alignment files accurately link to this report and retain the paper-reproduction limits.
- **SC-006**: Notebook 04 reports descriptor alignment and retrieval on a fixed held-out subset; it does not report training-set recall as evaluation.

## Assumptions

- The local CPU environment can run these bounded tutorial examples. A CPU-only result is sufficient for the stated local evidence.
- CIFAR-10 and the selected ResNet-18 weights may need public downloads for notebook 04. The run report records whether the sources were cached or fetched.
- The existing Gymnasium environments and notebook examples remain educational analogues. This feature does not claim to reproduce a paper benchmark.
- The source tree and notebook layout remain unchanged. Generated data and checkpoints stay in ignored or temporary paths.

## Scope

### In Scope

- Reproducibility controls and matched evaluation seeds for notebooks 01 and 04–08.
- Clean-kernel CPU execution and one local evidence report.
- Updates to curriculum coverage, project alignment, and remaining-work records.

### Out of Scope

- Reproducing V-JEPA, LeJEPA, VL-JEPA, GRASP, or physical-robot paper benchmarks.
- Replacing the synthetic tasks or pretrained visual backbone in the existing lessons.
- Adding new libraries, robot hardware, or private data.

## Open Questions

None. The scope is limited to reproducible local evidence for the existing lesson behavior.
