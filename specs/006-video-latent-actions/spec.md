# Feature Specification: Continuous Video-Derived Latent Actions

**Feature Branch**: `006-video-latent-actions`

**Created**: 2026-10-14

**Status**: Implemented locally; paper reproduction remains open

**Input**: Project vision task T013: compare continuous constrained latent actions derived from video with the current discrete latent-action lesson, and measure camera and embodiment transfer limits.

## User Scenarios & Testing

### User Story 1 - Compare continuous and discrete latent actions (Priority: P1)

A learner wants to see how an inverse dynamics model can infer a latent action from two video frames, and how continuous constrained actions differ from discrete codes.

**Why this priority**: The comparison is the main teaching outcome and extends the current discrete toy lesson.

**Independent Test**: Run the notebook from top to bottom. Confirm that the same generated frame pairs train both methods and that the notebook reports per-seed held-out prediction error and latent-use summaries.

**Acceptance Scenarios**:

1. **Given** a generated clip set, **When** the notebook infers actions from adjacent frames, **Then** it reports bounded continuous latents and discrete code assignments without using generator action labels as model inputs.
2. **Given** both methods train, **When** the notebook evaluates them, **Then** it reports each seed and aggregate results on the same held-out clips.

### User Story 2 - Inspect camera and object transfer (Priority: P2)

A learner wants to test whether an inferred action can be applied to a different object or camera view.

**Why this priority**: In-the-wild videos have different cameras and embodiments. The paper reports that learned actions can transfer between objects but are mainly localized relative to the camera.

**Independent Test**: Apply actions inferred from a source clip to matched target clips with a different shape and a different camera transform. Report the change in target-frame prediction error for each transfer.

**Acceptance Scenarios**:

1. **Given** a source and target clip with the same generated motion, **When** an inferred latent is applied to the target, **Then** the notebook reports transfer error for same-view, new-shape, and new-camera cases.
2. **Given** the toy transfer result, **When** a learner reads the interpretation, **Then** the notebook states that generated shapes and camera transforms do not establish transfer across real objects, embodiments, or video sources.

### User Story 3 - Separate tutorial evidence from paper evidence (Priority: P3)

A learner wants to understand how the lesson relates to the cited research and what is needed for a paper reproduction.

**Why this priority**: A small generated dataset and model cannot validate the paper's in-the-wild findings.

**Independent Test**: Read the paper context and local run report. Confirm that the model, data, training scale, measured results, and missing reproduction resources are separated.

**Acceptance Scenarios**:

1. **Given** the cited paper, **When** the notebook explains its setup, **Then** it links to the version 2 paper and reports its frozen V-JEPA 2-L encoder, 128-dimensional actions, YoutubeTemporal-1B training, and reported compute scale.
2. **Given** local results, **When** the report states them, **Then** it calls the work a synthetic analogue, not a paper reproduction or robot validation.

## Edge Cases

- A generated frame has no visible object pixels; feature extraction must fail with a clear message.
- A camera transform moves the object outside the frame; the notebook must stop with a clear error before training rather than silently use a cropped object.
- A model produces a non-finite latent or prediction; that run must be marked failed and excluded from aggregate metrics with the reason shown.
- One method fails or does not finish for a seed; the notebook must retain the missing-run record and must not compare unequal run counts without stating the limit.
- The optional display or plotting backend is unavailable; metric computation must still run and the learner must get a clear setup message.

## Requirements

### Functional Requirements

- **FR-001**: The notebook MUST explain inverse dynamics, forward prediction, constrained continuous actions, and discrete code assignments with a link to the primary paper.
- **FR-002**: The notebook MUST create a deterministic local set of short RGB clips with at least two object shapes and two camera views. It MUST state that these are generated data, not real video or paper data.
- **FR-003**: The notebook MUST derive its model inputs and targets from the generated frames with a documented fixed image-feature extraction step. Generator action labels MUST be held out from latent inference and model training.
- **FR-004**: The notebook MUST compare a bounded continuous latent-action model with a discrete latent-code baseline on identical train and held-out frame pairs, using fixed training seeds `[7, 17, 27, 37, 47]` and data seed `20261014`.
- **FR-005**: The notebook MUST report per-seed one-step mean squared error, averaged over frame-feature dimensions and held-out transitions, plus mean, median, standard deviation, and an uncertain 95% interval across five seeds for each method.
- **FR-006**: The notebook MUST measure action transfer to a new object shape and a changed camera view on matched generated motions. It MUST report the same mean squared error for each transfer setting separately.
- **FR-007**: The notebook MUST show continuous latent bounds and use or sparsity summaries, and discrete code usage. It MUST mark non-finite or incomplete runs.
- **FR-008**: The notebook MUST report software versions, device, data-generation seed, model seeds, sample counts, and training settings.
- **FR-009**: The notebook and result report MUST say that the experiment is a synthetic teaching analogue. It MUST NOT claim that it reproduces the paper, validates real-video transfer, or establishes physical robot performance.
- **FR-010**: The notebook MUST state that the paper uses frozen V-JEPA 2-L features, YoutubeTemporal-1B video, a 128-dimensional default latent action, and a reported 64-H100 training run of about 12 hours; these are paper-reported facts, not local measurements.
- **FR-011**: The notebook and report MUST state whether first-party code, checkpoint, and data access were confirmed during the dated source review. They MUST keep exact paper-reproduction work open if a required source is unavailable.

### Key Entities

- **Generated clip**: A short deterministic sequence of rendered frames, with shape, camera transform, and split metadata.
- **Frame feature**: A fixed, documented summary derived from one generated frame.
- **Latent action**: A continuous vector or discrete code inferred from a pair of frame features.
- **Transfer case**: A matched motion applied across the same view, another shape, or another camera transform.
- **Method run**: A continuous or discrete model run with seed, settings, predictions, and metrics.
- **Result summary**: Per-seed errors, transfer errors, latent-use measures, environment, and evidence limits.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A complete CPU run creates valid clips for all declared shapes and camera views. If a frame is invalid or cropped, the notebook stops before training with a clear error.
- **SC-002**: Both methods complete all five fixed-seed runs on the same held-out clips, or every failed run is reported with its reason.
- **SC-003**: The report includes per-seed and aggregate one-step prediction error for same-view, new-shape, and new-camera transfer cases.
- **SC-004**: The notebook shows finite continuous actions within the stated bounds and the discrete code-use distribution.
- **SC-005**: The notebook can run from top to bottom with the project environment on CPU without external data or model downloads.
- **SC-006**: No local result is described as a paper reproduction, real-video transfer result, or robot result.

## Assumptions

- Use generated short clips so the lesson does not depend on external video access, dataset terms, or a large checkpoint.
- Use a fixed, simple image-feature extractor and state clearly that it is not a learned JEPA encoder.
- Use a small model and data budget suitable for a local CPU run. This is a teaching comparison, not a scaled reproduction attempt.
- The primary paper source is its version 2 arXiv HTML dated 2026-01-20. A first-party code or checkpoint source was not confirmed in that paper source review.
- Do not add generated data, downloaded videos, checkpoints, or notebook execution outputs to version control.
