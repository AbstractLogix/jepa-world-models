# Feature Specification: Object-centric LeJEPA Lesson

**Feature Branch**: `008-object-centric-lejepa`

**Created**: 2026-10-14

**Status**: Local study complete; exact paper benchmark reproduction remains open

**Input**: User request to add an object-centric LeJEPA lesson and try to validate research findings.

## User Scenarios & Testing

### User Story 1 - Compare image and object representations (Priority: P1)

A learner compares image-level LeJEPA with a mask-pooled object-level variant on a small, public image dataset. The learner can see the mask source, what the masks supervise, how the two views are formed, and how one downstream task is measured.

**Why this priority**: This is the open object-centric lesson in the project roadmap. A clear comparison lets learners inspect the effect of object masks without confusing it with the paper's COCO benchmark.

**Independent Test**: Run `notebooks/13_object_centric_lejepa.ipynb` from a clean kernel with the stated data and hardware. Confirm that both arms finish five seeds and report the same held-out metric.

**Acceptance Scenarios**:

1. **Given** the Oxford-IIIT Pet trainval and test splits, **When** the learner runs the notebook, **Then** it reports the official data split, trimap source, mask rule, and dataset license before training.
2. **Given** the same model, image subset, optimizer, and training budget, **When** the image-level and mask-pooled arms train, **Then** the notebook reports all five seed results and balanced test accuracy for the same frozen linear-probe protocol.
3. **Given** any unavailable download, non-finite run, or invalid mask, **When** the run stops, **Then** the notebook preserves the failure and does not present a partial comparison as a result.
4. **Given** an official test image whose trimap has no pet or outline pixels, **When** the probe is evaluated, **Then** both methods use global-average pooling for that example and report the fallback count while retaining the full official test split.

### User Story 2 - Separate the local study from the paper result (Priority: P1)

A learner or contributor can inspect the paper's exact finding, current method description, official-code status, and the differences in the local study.

**Why this priority**: The paper's main result uses COCO and multiple training components. The local study uses a different dataset and a smaller model. The boundary must be visible wherever results are summarized.

**Independent Test**: Compare the notebook and run report with the paper's latest arXiv version and the reproduction matrix. Confirm they label local measurements as an exploratory transfer study, not a reproduction of the paper's benchmark.

**Acceptance Scenarios**:

1. **Given** the latest paper version, **When** the report describes its mask source, **Then** it says the main COCO experiment uses human ground-truth instance masks and identifies SAM proposals as a separate mask-source ablation.
2. **Given** no linked first-party object-centric code in the paper, **When** the local implementation is described, **Then** it states the implementation and objective components that are missing or simplified.
3. **Given** a local result that differs from the paper's setup, **When** the project matrix is updated, **Then** the exact COCO benchmark remains open and the local result is kept in a separate row or status field.

### Edge Cases

- Some official trimaps contain only background pixels. Exclude such trainval examples before balanced subset sampling. Keep all official test examples and use the same global-average pooling fallback for both methods when a test trimap has no foreground. Record both counts; do not label fallback features as mask-pooled.
- Oxford-IIIT Pet class counts may not support the planned balanced subset; use the fixed seed and the same per-class cap for both methods, then report the actual counts.
- One seed may fail or return a non-finite metric; retain its run record and do not silently replace it.
- CPU-only execution may be too slow for the five-seed study; the notebook must show its estimated settings before training and let a learner reduce the run only as an exploratory run.
- The dataset is about 800 MB and uses a Creative Commons Attribution-ShareAlike 4.0 license; the notebook must not commit data or imply that image copyright moved to the project.

## Requirements

### Functional Requirements

- **FR-001**: The lesson MUST compare a whole-image LeJEPA objective with a mask-pooled object-level LeJEPA variant under the same encoder, data split, optimizer, and run budget.
- **FR-002**: The lesson MUST identify Oxford-IIIT Pet trimaps as human-provided foreground, background, and boundary annotations and state how the boundary pixels form the foreground mask.
- **FR-003**: The lesson MUST use the official trainval and test splits. Breed labels MUST NOT enter the self-supervised training loss.
- **FR-004**: The downstream task MUST be frozen-feature pet-breed classification with a linear probe, evaluated by balanced accuracy on the official test split.
- **FR-005**: The comparison MUST use five training seeds, a fixed balanced pretraining subset with nonempty resized masks, the same full official test split, and report every run, the mean, median, and 95% interval. A test image with no foreground trimap MUST use global-average pooling for both methods, and the report MUST state the fallback count.
- **FR-006**: The notebook and report MUST record data and mask provenance, dataset license, code and model scale, optimizer, training budget, software versions, device, run time, and failures.
- **FR-007**: The notebook and report MUST label the result as an exploratory transfer study. They MUST NOT claim to reproduce the paper's COCO, DAVIS, ADE20k, NAVI, or ImageNet result.
- **FR-008**: The research note and roadmap MUST describe the current arXiv v2 paper accurately: ground-truth instance masks are used for the main experiment; SAM and superpixels are separate mask ablations; the paper says its object-centric training and SAM-mask code will be released.
- **FR-009**: The reproduction matrix MUST keep the paper's selected benchmark claim open until first-party code or a verified independent reproduction, COCO data, compute, and a paper-matched downstream evaluation are available.
- **FR-010**: Downloaded images, annotations, checkpoints, and notebook outputs MUST remain outside version control.

### Key Entities

- **View**: An augmented image and its aligned trimap, used as one input to a training arm.
- **Trimap**: The Oxford-IIIT Pet pixel annotation with pet, background, and outline values.
- **Representation**: A frozen feature vector pooled over the whole image or the annotated pet region.
- **Probe**: A linear classifier fitted on frozen trainval representations and evaluated on frozen test representations.
- **Run record**: A seed-specific outcome with configuration, metric, run time, device, and error state.
- **Paper claim**: The cited benchmark, metric, paper value, source version, and local reproduction status.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The notebook completes five seeds for both methods on the fixed local dataset, or preserves a clear failed-run record for each seed that does not finish.
- **SC-002**: Both methods use identical probe examples and evaluation code, and produce finite balanced-accuracy values on the full official held-out test split. The same global-pooling fallback is used for both methods when a test trimap has no foreground.
- **SC-003**: A reader can identify the source and role of every mask and all changes from the paper's model, data, objective, training budget, and evaluation task without searching code.
- **SC-004**: The paper reproduction matrix does not label the Oxford-IIIT Pet study as a completed reproduction of the paper's benchmark.

## Assumptions

- The current paper source is [arXiv v2](https://arxiv.org/abs/2607.02404), dated 2026-10-05. Its HTML reproducibility statement says that object-centric training and SAM-mask code will be released publicly; the source review did not find a linked first-party object-centric repository.
- The source dataset is [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), available under CC BY-SA 4.0. The dataset has breed labels and pixel-level trimap annotations.
- A small CNN, reduced image size, limited subset, and fewer epochs are acceptable for an exploratory transfer study. They do not reproduce the paper's ViT/COCO recipe or benchmark claim.
- The local study will use the paper's image-level versus object-level comparison as motivation. It will not implement the paper's full two-space objective, instance-separating loss, or copy-paste recipe unless a later, separately specified feature justifies that work.

## Scope

### In Scope

- One runnable notebook for a bounded image-level versus mask-pooled comparison.
- One downstream task, a run report, and updates to curriculum, roadmap, project alignment, and reproduction tracking.
- A dated research note that records paper version, reported values, source status, data, model, compute, and local deviations.

### Out of Scope

- Reproducing any COCO-trained paper benchmark or all paper ablations.
- Claiming that the local study validates the paper's findings.
- Adding a new package dependency, storing downloaded data, or adding an automatic dataset fetch outside the notebook.

## Open Questions

None. The local dataset and method limits are bounded by the current source and the project's existing evaluation protocol.
