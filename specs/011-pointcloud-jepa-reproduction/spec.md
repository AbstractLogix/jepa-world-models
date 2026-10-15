# Feature Specification: Point-Cloud JEPA Push-T Reproduction

**Feature Branch**: `011-pointcloud-jepa-reproduction`  
**Created**: 2026-10-14  
**Status**: Draft  
**Input**: Reproduce one reported point-cloud planning result with the paper's released Push-T assets and evaluation protocol.

## User Scenarios & Testing

### User Story 1 - Re-run the matched Push-T comparison (Priority: P1)

A learner can run the released Point-LeWM and Point-Delta-JEPA checkpoints on the same Push-T start-goal windows and inspect all per-seed results.

**Why this priority**: The comparison directly tests the paper's claim that the point-cloud model's planning result depends on its anti-collapse objective. The paper reports 83.6% ± 3.4 for Point-LeWM and 70.8% ± 7.7 for Point-Delta-JEPA on Push-T.

**Independent Test**: On the stated CUDA host, run both released checkpoints for the ten paper seeds and 50 episodes per seed. Confirm that each run uses the official v1 dataset, the same seed list, and the official Push-T planner settings.

**Acceptance Scenarios**:

1. **Given** the official code, v1 Push-T data, and both released checkpoints, **When** the two methods run for every paper seed, **Then** each method records 500 episode outcomes and ten per-seed success rates.
2. **Given** the two per-seed result lists, **When** the summary is calculated, **Then** it reports the mean and standard deviation across the ten seed rates and labels each mean inside or outside the paper's reported one-standard-deviation range.
3. **Given** the encoder resamples points during planning, **When** a seed produces a different result from the authors' stored value, **Then** the report preserves the local output and records this source of run variation.

### User Story 2 - Keep the reproduction auditable (Priority: P1)

A contributor can trace every local result to the code, checkpoint, dataset version, configuration, seed, and compute host used for the run.

**Why this priority**: The repository has a corrected v2 dataset, while the released checkpoints and reported paper numbers use v1. A version mismatch would change the experiment.

**Independent Test**: Compare the run report with the command outputs, asset hashes, and the official paper protocol.

**Acceptance Scenarios**:

1. **Given** an asset whose revision or SHA-256 does not match this specification, **When** the run is prepared, **Then** the evaluation stops before it produces a reproduction claim.
2. **Given** a host without CUDA 12 and 24 GB of GPU memory, **When** the preflight runs, **Then** it records the hardware blocker and does not download the large dataset.

### Edge Cases

- Missing or mismatched v1 data or checkpoint files stop the run and produce no success claim.
- A CUDA, EGL, or compiled-kernel failure is recorded with the stage and error; no partial seed sweep is summarized as the full paper result.
- Planning resamples points, so repeated values may differ slightly even when the same evaluation seed selects the same start-goal windows.
- A video-rendering failure does not invalidate numeric results when video output is disabled as documented by the official evaluator.

## Requirements

### Functional Requirements

- **FR-001**: The evaluation MUST use the official `fafraob/point-lewm` code at commit `1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424` and record that revision.
- **FR-002**: The evaluation MUST use the official Push-T dataset v1 at revision `d3391d60371b6e9cbcd1c5737bdfc87b4af309ed`, archive SHA-256 `80501fbca03fce36273be84086f3b0b6148b77fa4c5831d8085be7cdd91c1171`, and the released Point-LeWM and Point-Delta-JEPA checkpoints at model revision `664278a4c0b4c5f607d0993744986b2d10b4168e`.
- **FR-003**: The evaluation MUST verify Point-LeWM weights SHA-256 `42e1e85134497c8a2af0b9620d72c6421de767f1b85ee9778473c83292f81710` and Point-Delta-JEPA weights SHA-256 `4a4315fdfae6b0ce15868fbb2bb4d80a6ebeda6c2d3f0a7c1f71cb8f1b36a237` before use.
- **FR-004**: The evaluation MUST use the official Push-T LiDAR configuration, closed-loop CEM with 300 samples, 30 iterations, horizon 5, frame skip 5, goal offset 25, and a 50-environment-step budget.
- **FR-005**: Both methods MUST use seeds `1, 25, 35, 36, 40, 42, 65, 80, 83, 99`, with 50 episodes per seed and the same start-goal windows for each paired seed.
- **FR-006**: The report MUST include all 1,000 episode outcomes, per-seed success rates, the mean and standard deviation across the ten seed rates, and the exact run configuration.
- **FR-007**: The report MUST compare the local means with the paper values of 83.6% ± 3.4 for Point-LeWM and 70.8% ± 7.7 for Point-Delta-JEPA. The one-standard-deviation range is a descriptive check, not a statistical test.
- **FR-008**: The run MUST record the GPU model and memory, CUDA driver, Python and PyTorch versions, Pixi lock revision, run time, and any deviations or errors.
- **FR-009**: Dataset files, checkpoints, upstream checkout, logs, and generated videos MUST stay outside version control. The project MUST NOT add the upstream environment as a dependency.
- **FR-010**: The run MUST stop before downloading the dataset unless the host meets the official Linux, CUDA 12, EGL, and 24 GB GPU requirements.

### Key Entities

- **Evaluation asset**: The code, dataset, or checkpoint with its repository, revision, license, byte size, and SHA-256.
- **Evaluation seed**: One paper seed that selects 50 shared start-goal windows and produces one success rate per method.
- **Episode outcome**: One Push-T attempt with its seed, method, success flag, and episode index.
- **Reproduction report**: Asset provenance, compute details, per-episode and per-seed results, summary comparison, deviations, and claim limits.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Two methods complete all ten seeds with exactly 50 episodes per seed, for 500 recorded episodes per method.
- **SC-002**: Every paired method result uses the same evaluation seed and the same 50 start-goal windows.
- **SC-003**: The report includes both method means, standard deviations across seeds, paper target ranges, and all raw per-seed and per-episode outcomes.
- **SC-004**: The report states whether each local mean falls inside the paper's one-standard-deviation range and does not present that check as a significance test.
- **SC-005**: All run assets match the specified revisions and hashes, and all generated files remain outside version control.

## Assumptions

- The paper's v1 data and checkpoint assets are the intended target; the October 2026 v2 dataset is not a substitute because it changes the sensor-view images and corrects Push-T scan replay.
- This feature evaluates released checkpoints. It does not train models or reproduce the paper's training runs.
- The report can omit videos when using the official `++output.video=false` option; the numeric protocol remains unchanged.
- The current project host does not meet the GPU requirement. A compatible host is needed before the evaluation can start.

## Scope

### In Scope

- One paired Push-T planning comparison between Point-LeWM and Point-Delta-JEPA.
- The official data, checkpoints, CEM settings, seeds, and episode count.
- A local evidence report and updates to curriculum and reproduction status.

### Out of Scope

- Training either model, running the other environments, reproducing the target-to-latent results, or reproducing the sensor-degradation tables.
- Adding or vendoring the upstream implementation, model assets, or dataset in this repository.
- Claims about physical robots or transfer outside the four simulated scenes.

## Open Questions

None. Evaluation is gated on access to a host that meets the official resource requirements.
