# Feature Specification: V-JEPA 2.1 Dense Video Features Tutorial

**Feature Branch**: `002-vjepa-21-dense-features`

**Created**: 2026-10-14

**Status**: Implemented; the official checkpoint CPU path was repeated in a fresh kernel on 2026-10-15. See [the local run report](../../docs/evidence/vjepa21-local-run.md).

**Input**: User goal to deliver the project's future tutorials with reproducible local evidence. This feature delivers the roadmap's V-JEPA 2.1 dense-feature lesson.

## User Scenarios & Testing

### User Story 1 - Inspect dense video features (Priority: P1)

A learner wants to see how a released V-JEPA 2.1 model represents a short video. They need a small, self-contained input and a clear view of frame, time, and patch features.

**Why this priority**: Before this feature, the curriculum had no released video checkpoint. This lesson adds a direct path from toy examples to features from a current video model.

**Independent Test**: Run the lesson with the documented official checkpoint and generated clip. Confirm the feature grid has the expected temporal and spatial dimensions and that the visual output maps back to the input frames.

**Acceptance Scenarios**:

1. **Given** a learner has installed the documented optional dependencies, **When** they opt in to load the smallest released V-JEPA 2.1 checkpoint, **Then** the lesson records its source and version before downloading and loads its encoder.
2. **Given** the generated clip is processed, **When** the learner inspects the output, **Then** the lesson shows a dense feature grid for each encoded time step and identifies its frame, patch, and channel dimensions.

### User Story 2 - Compare temporal patch correspondence (Priority: P2)

A learner wants to examine whether dense patch features can match a moving object across time. They need known motion and a simple baseline so they can interpret the result without treating a toy clip as a benchmark.

**Why this priority**: A correspondence exercise makes temporal consistency visible and measurable while avoiding a large external video dataset.

**Independent Test**: Run the correspondence section on the deterministic generated clip. Confirm it reports the target location, retrieved location, similarity, and the same measurement for a named pixel-feature baseline.

**Acceptance Scenarios**:

1. **Given** the clip generator's known object path, **When** a learner compares features from two time groups, **Then** the lesson reports the retrieved patch and its distance from the known target patch.
2. **Given** the model or baseline misses the target, **When** the lesson displays the result, **Then** it reports the miss as measured and does not claim that the model learned correspondence from this example.

### User Story 3 - Run within local resource limits (Priority: P3)

A learner wants to use the lesson on a local machine without robot hardware or a private dataset. They need to know the checkpoint size, storage location, device path, and what can run without downloading weights.

**Why this priority**: The official smallest checkpoint is still large. Clear opt-in and resource guidance prevent surprise downloads and make the tutorial usable on CPU-only systems.

**Independent Test**: Start from a clean environment. The generated video and plotting sections run without network access. Loading the official checkpoint is a separate, explicit action and has both CPU and available-accelerator paths.

**Acceptance Scenarios**:

1. **Given** a learner does not opt in to the checkpoint, **When** they run the setup and clip-generation sections, **Then** no video or model weights are downloaded.
2. **Given** no accelerator is available, **When** a learner opts in and has sufficient memory and time, **Then** the lesson uses its CPU path and records the device and elapsed time.

## Edge Cases

- The learner has no network access or declines the large checkpoint download.
- The checkpoint source, code revision, or checkpoint terms change or become unavailable.
- The host has too little memory or time for CPU inference.
- The official loading API changes or its default checkpoint URL is unavailable.
- The generated clip dimensions do not divide evenly by the patch or tubelet size.
- The retrieved patch does not match the known object location.
- A model or preprocessing change alters feature dimensions or output order.

## Requirements

### Functional Requirements

- **FR-001**: The lesson MUST state its learner outcome, prerequisites, expected outputs, and that it is an official-checkpoint feature inspection, not a paper reproduction.
- **FR-002**: The lesson MUST use a deterministic video clip that it creates locally and MUST NOT require an external video dataset for its core path.
- **FR-003**: The lesson MUST use a named official V-JEPA 2.1 checkpoint and record the model source, checkpoint identifier, code revision, file size, and the checkpoint-license status before loading it.
- **FR-004**: Checkpoint download MUST be an explicit learner action. Generated clips, weights, and derived features MUST remain outside version control.
- **FR-005**: The lesson MUST show dense spatial and temporal features and explain how tensor dimensions map to the clip and patch grid.
- **FR-006**: The lesson MUST measure a known moving-object patch correspondence and compare it with a named pixel-feature baseline. It MUST report the measured values without a pass threshold or broad model claim.
- **FR-007**: The lesson MUST state the device, input shape, model revision, preprocessing, elapsed time, and available memory or accelerator for each local feature run.
- **FR-008**: The lesson MUST include a no-download path for clip generation and visualization, and a CPU inference path for the official checkpoint when local memory and time allow.
- **FR-009**: Documentation MUST separate official paper results, local checkpoint measurements, and generated-clip illustrations.

### Key Entities

- **Video clip**: Sixteen deterministic RGB frames at 384×384 pixels, with recorded frame dimensions and known object positions. Sixteen frames map to eight temporal tubelets and a 24×24 spatial patch grid for the selected model configuration.
- **Checkpoint record**: Official source, identifier, code revision, file size, terms status, and local cache location.
- **Feature grid**: Encoder output associated with a clip, time grouping, patch grid, device, and preprocessing record.
- **Correspondence result**: Query patch, known target patch, retrieved patch, similarity, baseline result, and run conditions.

## Success Criteria

### Measurable Outcomes

- **SC-001**: From a clean environment with the documented optional dependencies, a learner can generate and visualize the clip without network access.
- **SC-002**: With the official checkpoint available, one documented CPU run produces finite feature values and a feature grid whose dimensions agree with the 16-frame, 384-pixel input, 16-pixel patch size, and 2-frame tubelet size.
- **SC-003**: One run reports both V-JEPA 2.1 and pixel-baseline correspondence results for the same known object path.
- **SC-004**: Every local feature result includes the conditions listed in FR-007; no paper-reported result is labeled as local.
- **SC-005**: The README, curriculum coverage map, roadmap, and this feature's specification agree on the lesson's source, outcome, resource needs, and reproduction boundary.

## Assumptions

- The smallest official V-JEPA 2.1 release (ViT-B/16 at 384 pixels) is the selected teaching model because it is the smallest released model in the official repository.
- The public Hub entrypoint fixes the model input at 384 pixels. The lesson uses that image size and reduces the temporal input to 16 frames.
- The clip is synthetic and deterministic to avoid third-party video licensing and download requirements. It is a diagnostic input, not a substitute for a real-video benchmark.
- The official checkpoint is optional and is not committed. The official code repository lists MIT and Apache-2.0 licenses for code, but does not state a separate checkpoint license in its README; the lesson will record that uncertainty and link the source terms before download.
- CPU execution may be slow and requires enough memory for the checkpoint and model. The lesson will not promise a fixed runtime across machines.
- The feature correspondence exercise is illustrative. No expected score or research-level conclusion is set.
