# Paper-aligned JEPA objectives

**Feature Branch**: `007-paper-aligned-losses`

**Created**: 2026-10-14

**Status**: Implemented locally; paper benchmark reproductions remain open

**Input**: Project vision task T014: compare the current teaching losses with paper-aligned LeJEPA and Rectified LpJEPA runs, and record data, model scale, metrics, and seeds in notebooks 02 and 03.

## User Scenarios & Testing

### User Story 1 - Compare LeJEPA losses (Priority: P1)

A learner wants to see how the current LeJEPA teaching loss compares with the paper's SIGReg objective under a small, fixed CIFAR-10 setup.

**Why this priority**: Notebook 02 is the first objective lesson. It must show the paper's method and the limits of a small local run before the project can compare sparse objectives.

**Independent Test**: Run notebook 02 on CPU from a clean kernel. It reports each planned seed, a frozen linear-probe score, and the loss and model settings for both methods.

**Acceptance Scenarios**:

1. **Given** the same CIFAR-10 training subset and model family, **When** the learner runs the current teaching version and the paper-aligned LeJEPA version for each planned seed, **Then** the notebook reports every run with its settings and outcome.
2. **Given** the paper's larger ImageNet results, **When** the notebook reports local scores, **Then** it labels local measurements separately and says that the run is not a paper reproduction.

### User Story 2 - Compare Rectified LpJEPA losses (Priority: P1)

A learner wants to compare the existing RDM-style teaching proxy with the paper's distribution-matching objective and see whether the output features are sparse and useful for classification in the local setup.

**Why this priority**: The current lesson calls a sparsity and entropy proxy RDM-style. A paper-aligned run must make the distinction visible and measure both sparsity and task information.

**Independent Test**: Run notebook 03 on CPU from a clean kernel. It reports each planned seed, a frozen linear-probe score, and the defined sparsity metrics for both methods.

**Acceptance Scenarios**:

1. **Given** the same data split, backbone, and training budget, **When** the learner compares the teaching proxy with RDMReg, **Then** each arm reports its per-seed encoder and projector scores, sparsity, and failures.
2. **Given** a lower sparsity score, **When** the notebook explains the result, **Then** it uses the paper's metric definition and does not claim the result matches ImageNet-100.

### Edge Cases

- A failed, non-finite, or interrupted run remains in the results with its failure reason; it is not dropped from a summary.
- A constant or near-zero embedding is reported as collapse, and sparsity is not treated as useful if the linear probe is at chance.
- CIFAR-10 data or required Python packages are not available: the notebook gives the source and install path and stops with a clear error.
- The report has fewer than five completed independent seeds: classify the local work as exploratory and do not claim a general comparison.

## Requirements

### Functional Requirements

- **FR-001**: The notebooks MUST explain that the current LeJEPA lesson uses a reduced SIGReg sketch and that the current Rectified LpJEPA lesson uses a variance/covariance plus L1/entropy proxy, not the paper's RDMReg.
- **FR-002**: The paper-aligned LeJEPA run MUST implement the Epps-Pulley characteristic-function SIGReg objective, its isotropic-Gaussian target, and the paper's invariance-plus-SIGReg weighting. It MUST record any reduced projection count or view change.
- **FR-003**: The paper-aligned Rectified LpJEPA run MUST rectify projector outputs and match them to a named Rectified Generalized Gaussian target using sliced 2-Wasserstein RDMReg. It MUST record the target parameters and projection count.
- **FR-004**: Both notebook comparisons MUST use the same named CIFAR-10 source, fixed disjoint self-supervised, probe-training, and test subsets, one declared model scale per comparison, and the same training budget across paired methods.
- **FR-005**: Each comparison MUST use the project default of at least five fixed independent training seeds, report every run, and give mean, median, and an explicitly uncertain 95% interval for each primary metric. If the resource limit prevents five completed runs, it MUST be reported as exploratory.
- **FR-006**: Notebook 02 MUST report frozen linear-probe top-1 accuracy and embedding collapse/isotropy measurements. Notebook 03 MUST report frozen encoder and projector linear-probe top-1 accuracy, the paper-defined L1 and L0 sparsity metrics, and per-run failures.
- **FR-007**: The lesson text and report MUST separate paper-reported findings from locally measured results and MUST identify the local study as small-scale and not a reproduction of the papers' benchmark results.
- **FR-008**: The source review, local method, data, model scale, metrics, seeds, runtime, and limitations MUST be recorded in a run report and aligned with the roadmap, coverage map, reproduction matrix, and project vision task T014.

### Fixed local study

- Data: CIFAR-10 official train and test splits. Use 640 balanced images for self-supervised training, 1,000 disjoint balanced train images for the linear probe, and 1,000 balanced test images. The first 1,000-image execution exceeded the 900-second notebook limit, so the final local study reduces only the self-supervised subset to 64 images per class. Fix the selected indices with data seed `20261014`.
- Backbone: one small four-stage CNN with a 256-dimensional representation, about 455,000 parameters before the Rectified LpJEPA projector. Report the exact count from the built model.
- Training: 20 epochs, batch size 128, AdamW at `0.003`, weight decay `1e-5`, and training seeds `[7, 17, 27, 37, 47]`. Match the view setup within each pair: notebook 02 uses two global and four local views in both arms; notebook 03 uses two augmented views in both arms. LeJEPA recommends two global and eight local views, so notebook 02 uses fewer local views.
- LeJEPA arms: retain the notebook's 128-slice, `lambda=0.1` Epps-Pulley sketch as the teaching arm. Use 512 random projections, 17 quadrature points over `[-5, 5]`, per-view SIGReg, and `lambda=0.05` for the paper-aligned objective. Record that the paper recommends 1,024 slices and that this local run uses fewer.
- Rectified LpJEPA arms: retain the current variance/covariance plus L1/entropy helper losses as the teaching arm. Use a 256-wide, three-layer projector with final ReLU and the paper's Rectified Laplace target (`p=1`, `mu=0`, `sigma=1/sqrt(2)`), 512 random projections, and loss weights 25 (invariance) and 125 (RDMReg). Record the lower projection count and all optimizer, data, model, and epoch differences from the paper's ImageNet-100 setup.
- Evaluation: fit the existing frozen linear probe on the fixed labeled train and test subsets. Report top-1 accuracy, mean, median, and a two-sided 95% Student-t interval across the five training seeds. Also report feature standard-deviation and mean absolute off-diagonal covariance for LeJEPA, plus the paper-defined projector L1 and L0 sparsity metrics for Rectified LpJEPA.

### Key Entities

- **Objective comparison**: Teaching or paper-aligned loss, objective settings, backbone, projector, and training budget.
- **Data split**: CIFAR-10 source split, subset size, fixed indices, and role (self-supervised training, labeled probe training, or final evaluation).
- **Training run**: Objective, seed, outcome, elapsed time, and measured metrics.
- **Paper finding**: Source paper and version, reported claim, and its original data/model/compute scope.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Both notebooks show all five planned seeds for each objective, including any failed run.
- **SC-002**: Every reported local score can be traced to the named data split, model, objective, and training seed.
- **SC-003**: The Rectified LpJEPA comparison reports both L1 and L0 sparsity and a frozen linear-probe score for every completed seed.
- **SC-004**: The roadmap, coverage map, reproduction matrix, and project task agree that the local objective runs are not paper benchmark reproductions.

## Assumptions

- Use the official CIFAR-10 train and test data already required by the two lessons. Do not commit downloaded data, weights, checkpoints, or generated outputs.
- Use the fixed data and training settings above on CPU. This is a local objective study, not a scaled reproduction attempt of an ImageNet benchmark.
- Keep existing teaching proxies available so learners can compare their behavior with the paper-aligned objectives.
- Use no new package dependency. The project already supports PyTorch and TorchVision in its tutorial environment.
- Follow `docs/EVALUATION_PROTOCOL.md`; report five seeds or mark the result exploratory.
