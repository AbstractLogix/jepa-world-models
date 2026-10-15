# Feature Specification: VL-JEPA Real Label Embeddings

**Feature Branch**: `005-vljepa-real-labels`

**Created**: 2026-10-14

**Status**: Implemented locally; paper reproduction remains open

**Input**: Project vision task T012: use real caption or label embeddings and compare classification or retrieval with a baseline in a new advanced VL-JEPA lesson.

## User Scenarios & Testing

### User Story 1 - Predict real text embeddings (Priority: P1)

A learner wants to see how an image model can predict the embedding of a real text label. They run an advanced notebook that uses CIFAR-10 labels and a public text-embedding checkpoint.

**Why this priority**: This is the main concept in the lesson and replaces the synthetic target exercise with a real language target.

**Independent Test**: Run the notebook from top to bottom. Confirm that it creates one normalized text embedding for each class prompt, trains an image-to-text predictor, and ranks the ten candidate class prompts for each held-out image.

**Acceptance Scenarios**:

1. **Given** the listed optional dependencies and network access, **When** the learner runs the notebook, **Then** the notebook loads the pinned text checkpoint and official image weights and displays their sources, sizes, and license limits before training.
2. **Given** a held-out image, **When** the predictor runs, **Then** it compares the predicted text embedding with the ten fixed label embeddings and returns a ranked class list.

### User Story 2 - Compare with a supervised baseline (Priority: P2)

A learner wants to know whether predicting text embeddings performs differently from direct classification on the same visual features.

**Why this priority**: A baseline gives context to the local measurement and prevents an ungrounded claim about the embedding objective.

**Independent Test**: Run both methods on the same fixed train and held-out subsets. Confirm they use the same frozen image features and report top-1 accuracy for each of five initialization seeds.

**Acceptance Scenarios**:

1. **Given** a fixed balanced CIFAR-10 subset, **When** each method trains, **Then** both use the same training images, held-out images, and seed list.
2. **Given** the two methods finish, **When** the results are shown, **Then** per-seed top-1 accuracy and mean and standard deviation are reported, with the linear classifier identified as the baseline.

### User Story 3 - Understand what the result supports (Priority: P3)

A learner wants to separate the paper's claims from the small local experiment and understand what would be needed for a closer reproduction.

**Why this priority**: The local setup uses a small image dataset, class labels, and a much smaller model stack than the paper.

**Independent Test**: Read the paper and local result sections. Confirm that the paper setup, local measurements, data and checkpoint limits, and missing reproduction components are labeled separately.

**Acceptance Scenarios**:

1. **Given** paper-reported results, **When** the notebook summarizes them, **Then** it links the primary paper and names the paper's model, training data scale, and benchmark scope.
2. **Given** local measurements, **When** the report states results, **Then** it identifies the lesson as a VL-JEPA-inspired analogue and does not call it a paper reproduction or zero-shot classification.

### Edge Cases

- The pinned model or data source is unavailable; the notebook must stop with a clear download error and must not substitute a different checkpoint silently.
- CIFAR-10 is missing or incomplete; the learner must be told which official data source and download are needed.
- A sampled class has fewer examples than requested; the notebook must fail before training rather than silently make an unbalanced subset.
- A method produces non-finite embeddings or scores; the run must report the issue instead of adding the result to the summary.
- Network access is unavailable on the first run; the notebook must state that cached weights and data are needed for offline runs.

## Requirements

### Functional Requirements

- **FR-001**: The notebook MUST explain the paper's image encoder, predictor, text encoder, embedding-space objective, and optional text decoder, and link to the primary paper.
- **FR-002**: The notebook MUST use the ten official CIFAR-10 class labels as real text targets and MUST state that it uses class prompts rather than human-written captions.
- **FR-003**: The notebook MUST identify the text checkpoint by model ID and immutable revision, use the checkpoint's documented pooling and normalization, and record its source, license, and download size.
- **FR-004**: The notebook MUST use a named, frozen ImageNet-pretrained ResNet-18 image encoder with its documented preprocessing. It MUST record the weight source, file size, and any license or use terms that remain unclear.
- **FR-005**: The notebook MUST train an image-to-text-embedding predictor and select among the ten candidate label embeddings by similarity.
- **FR-006**: The notebook MUST compare the predictor with a linear classifier trained on the same frozen image features, training examples, held-out examples, and initialization seeds `[7, 17, 27, 37, 47]`.
- **FR-007**: The notebook MUST select exactly 100 images per class from the official CIFAR-10 training split and 100 per class from its official test split using data seed `20261014`. It MUST keep those splits disjoint.
- **FR-008**: The notebook MUST report per-seed top-1 accuracy, mean, median, standard deviation, and a 95% interval across the five training seeds for both methods. It MUST label those intervals uncertain because there are only five seeds. It MUST also report a paired per-image accuracy difference with a 95% class-stratified bootstrap interval on the fixed held-out sample, and class-wise results.
- **FR-009**: The notebook MUST record software versions, CPU/GPU, sample counts, data seed, model seeds, and training settings.
- **FR-010**: The notebook and result report MUST mark the experiment as an illustrative analogue, not a VL-JEPA reproduction, a zero-shot result, or a test of open-vocabulary generalization.
- **FR-011**: The notebook MUST state that CIFAR-10's official page requests a citation but does not state dataset license terms, and MUST not redistribute dataset or checkpoint files.
- **FR-012**: The notebook MUST state the first-run download sizes and make clear that network access is required unless data and checkpoints are already cached.

### Key Entities

- **Class prompt**: A fixed natural-language prompt for one CIFAR-10 class and its normalized text embedding.
- **Visual feature**: A frozen ResNet-18 representation for one CIFAR-10 image.
- **Evaluation sample**: A class-balanced training or held-out image, its class ID, and its fixed seed-derived selection.
- **Method run**: One predictor or linear-classifier training run with seed, settings, and held-out predictions.
- **Result summary**: Per-seed, mean, standard deviation, and class-wise top-1 measurements with environment and evidence limits.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A complete run processes exactly 1,000 training and 1,000 held-out images, with 100 images per class in each split and no overlap between the official train and test splits.
- **SC-002**: Both methods complete all five fixed-seed runs and report each run, mean, median, standard deviation, and uncertain 95% interval across runs.
- **SC-003**: A learner can inspect candidate prompt scores and identify the highest-scoring class for each held-out image.
- **SC-004**: The notebook and report identify all model and data sources, checkpoint revision, download sizes, access needs, and unknown license terms.
- **SC-005**: The full notebook runs without errors on CPU when the listed optional dependencies, data, and checkpoints are available.
- **SC-006**: No local result is described as a paper result, zero-shot result, or reproduction of VL-JEPA.

## Assumptions

- The lesson uses one class prompt per CIFAR-10 label, not image captions. CIFAR-10's official page describes 50,000 training images and 10,000 test images and gives a 163 MB Python archive; it does not list a license statement.
- The image encoder uses TorchVision's `ResNet18_Weights.IMAGENET1K_V1` (44.7 MB) and the text encoder uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, Apache-2.0, 90.9 MB model file) pinned to revision `ea78891063587eb050ed4166b20062eaf978037c`.
- The additional Transformers and Safetensors packages are optional and are installed separately from PyTorch. The CPU path is the supported local fallback.
- The scope is a small supervised analogue for learning the embedding-prediction mechanism. It does not train V-JEPA 2, Llama, EmbeddingGemma, bidirectional InfoNCE, a decoder, or the paper's video benchmarks.
- Dataset and model downloads are cached outside version control. The learner is responsible for confirming applicable terms before reuse or redistribution.
