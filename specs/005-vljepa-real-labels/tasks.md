# Tasks: VL-JEPA Real Label Embeddings

**Input**: Design documents from `specs/005-vljepa-real-labels/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`

**Tests**: No separate test suite was requested. Run the notebook because the feature requires local measurements and a complete execution record.

## Phase 1: Setup

**Purpose**: Add only the optional runtime needed for this lesson.

- [x] T001 Add the `vljepa` optional dependency extra for Transformers and Safetensors in `pyproject.toml` and regenerate the matching `poetry.lock` entries.
- [x] T002 Add notebook 11 to the learning path and explain its optional install and download needs in `README.md` and `CURRICULUM_GUIDE.md`.

## Phase 2: User Story 1 - Predict real text embeddings (Priority: P1)

**Goal**: Encode real CIFAR-10 class prompts and predict their text embeddings from frozen image features.

**Independent Test**: Run notebook 11 through inference. Confirm ten normalized prompt embeddings, balanced inputs, finite predicted embeddings, and ranked candidate labels.

- [x] T003 Create `notebooks/11_vl_jepa.ipynb` with lesson goals, prerequisites, primary paper context, data/checkpoint sources, sizes, terms, and analogue/reproduction limits.
- [x] T004 [US1] Load the pinned MiniLM model, create normalized class-prompt embeddings with the documented pooling, and assert the expected dimension in `notebooks/11_vl_jepa.ipynb`.
- [x] T005 [US1] Select the fixed balanced CIFAR-10 train/test subsets and extract frozen ResNet-18 features with official preprocessing in `notebooks/11_vl_jepa.ipynb`.
- [x] T006 [US1] Train the embedding predictor and rank held-out images against class prompt embeddings in `notebooks/11_vl_jepa.ipynb`.

## Phase 3: User Story 2 - Compare with a supervised baseline (Priority: P2)

**Goal**: Compare text-embedding prediction with a linear classifier on the same features and examples.

**Independent Test**: Run both methods with the five specified seeds and confirm complete per-seed, aggregate interval, paired per-image, and class-wise results.

- [x] T007 [US2] Train the linear-classifier baseline with the same frozen image features, data, and seeds in `notebooks/11_vl_jepa.ipynb`.
- [x] T008 [US2] Report raw five-seed results, uncertain 95% t intervals across seeds, class-stratified paired per-image bootstrap interval, class-wise accuracy, and held-out prompt scores in `notebooks/11_vl_jepa.ipynb`.

## Phase 4: User Story 3 - Interpret evidence honestly (Priority: P3)

**Goal**: Keep the paper's scale and findings separate from the local analogue.

**Independent Test**: Confirm that notebook and report distinguish paper evidence, local measurements, and the missing resources for exact reproduction.

- [x] T009 [US3] Add the paper findings, the local method and limits, and measured environment details to `notebooks/11_vl_jepa.ipynb`.
- [x] T010 [US3] Record the completed run in `docs/reproductions/vl-jepa.md` and align the VL-JEPA row and status notes in `docs/ROADMAP.md`, `docs/REPRODUCTION_MATRIX.md`, and `docs/CURRICULUM_COVERAGE.md`.
- [x] T011 [US3] Mark project task T012 complete only after the report and notebook evidence are complete in `specs/001-project-vision/tasks.md`.

## Phase 5: Run and alignment checks

**Purpose**: Verify the runnable lesson and cross-artifact consistency.

- [x] T012 Parse the notebook and execute it top to bottom on CPU using `specs/005-vljepa-real-labels/quickstart.md`; record the result in `docs/reproductions/vl-jepa.md`.
- [x] T013 Run Spec Kit analysis and convergence for this feature. Resolve artifact inconsistencies and append only actual unbuilt work to this task list; no remaining feature work was found.

## Dependencies and Execution Order

- T001 and T002 precede the learner-facing run.
- T003 defines the notebook's source, evidence, and download contract. T004 and T005 prepare the text and image representations before T006 trains the predictor.
- T007 uses the same features and examples as T006. T008 depends on both methods.
- T009 and T010 depend on paper review and measured output. T011 depends on the completed report. T012 verifies the finished notebook and report. T013 follows implementation and verification.
- No tasks are parallelizable because notebook code is sequential and documentation depends on verified results.

## Implementation Strategy

Build the smallest complete notebook first. Compare both methods on the exact same frozen features and examples. Run the full notebook on CPU, write the evidence report from that run, update project tracking, then analyze and converge the feature artifacts.
