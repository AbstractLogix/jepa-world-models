# Feature Specification: Preflight LeWAM RoboTwin 2.0

**Feature Branch**: `027-lewam-robotwin-preflight`

**Created**: 2026-10-15

**Status**: Implemented

**Input**: Project goal to research future JEPA tutorials and reproduce selected paper findings with traceable evidence.

## User Scenarios & Testing

### User Story 1 - Decide whether a RoboTwin evaluation can start (Priority: P1)

A learner or reviewer needs to know whether the released LeWAM checkpoint can be evaluated on RoboTwin 2.0 before any source checkout, asset transfer, or run. The report must name the paper claim, the official artifacts, the local platform limits, and the gates that remain open.

**Why this priority**: The paper reports a 50-task result, while the official default covers 10,000 cases by calculation. A small fixed-case run is a safer first step, but only if source terms and the simulator runtime are understood.

**Independent Test**: Review the preflight report and linked roadmap entries. Confirm that the report distinguishes paper values from local results, states that no local evaluation ran, records the source revision and access/runtime gates, and gives a bounded restart path.

**Acceptance Scenarios**:

1. **Given** a reader sees the reported 92.28% average success, **When** they read this project's status, **Then** they can identify it as a paper-reported value with no local result.
2. **Given** the current host is WSL2 and lacks the CUDA toolkit and Vulkan tools, **When** a reviewer checks readiness, **Then** the report says not to start the simulator evaluation on this host.
3. **Given** the code repository has no declared project license, **When** a reviewer checks source access, **Then** the report records the unresolved terms and does not claim that the encoder's license covers the code or predictor.
4. **Given** the evaluation candidate is listed in the roadmap and matrix, **When** a reader checks project alignment, **Then** all records link to the same preflight and keep it distinct from LeWAM Push-T.

### Edge Cases

- The paper or official repository changes after the reviewed revisions.
- A checkpoint license does not cover the predictor, code, or simulator assets.
- A fixed case list covers only one task or setting and cannot support the paper's 50-task average.
- Adding Vulkan or CUDA packages inside WSL does not make WSL GPU simulation or rendering supported.

## Requirements

### Functional Requirements

- **FR-001**: The project MUST provide a dated preflight report for released-checkpoint evaluation of LeWAM on RoboTwin 2.0.
- **FR-002**: The report MUST identify the paper-reported claim, official source revision, checkpoint contents, evaluation scale, and required environment without presenting paper values as local measurements.
- **FR-003**: The report MUST record the current host's GPU, WSL platform, missing runtime components, storage state, and license status, with a source for each external requirement.
- **FR-004**: The report MUST state that no local code checkout, simulator asset, checkpoint transfer, installation, or evaluation occurred for this preflight.
- **FR-005**: The report MUST define a fixed-case restart path and the conditions that must pass before evaluation. A single task result MUST NOT be presented as reproducing the paper's 50-task average.
- **FR-006**: The roadmap, reproduction matrix, project alignment, remaining-work record, and project task list MUST point to the same status and keep RoboTwin separate from LeWAM Push-T.

## Key Entities

- **Candidate**: A paper claim and its proposed released-checkpoint evaluation scope.
- **Source revision**: A pinned paper, code revision, model card, or simulator guide used to define the candidate.
- **Readiness gate**: A source-term, platform, runtime, storage, or case-list condition that must pass before a run.
- **Run status**: One of paper-reported, locally attempted, locally measured, or not run.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The preflight report records the reviewed paper and code revisions and links the official model and simulator sources.
- **SC-002**: The report lists every observed blocker and clearly records zero local evaluation results and zero transferred evaluation assets.
- **SC-003**: The project roadmap, reproduction matrix, alignment, and remaining-work records link to the same report and preserve the separate Push-T status.

## Assumptions

- The proposed evaluation is a released-checkpoint evaluation, not independent training.
- The first future run will use one named task and a declared fixed case list. It will be reported as a bounded check, not as a reproduction of the paper's aggregate result.
- No project license in repository metadata is treated as unresolved terms, not as permission to redistribute or vendor the code.
