# Feature Specification: Qantara Multi-Paradigm Inference Tutorial and Evaluation

**Feature Branch**: `[019-qantara-multiparadigm]`  
**Created**: 2026-10-15  
**Status**: CPU tutorial implemented; official evaluation is gated  
**Input**: Project vision: teach recent JEPA methods and make bounded attempts to validate their reported findings.

## User Scenarios & Testing

### User Story 1 - Understand three inference paths (Priority: P1)

As a learner, I want to see how one latent world model can support planning, behaviour-cloning action sampling, and video–inverse action recovery so that I can compare what each path needs at inference time.

**Why this priority**: This is Qantara's central tutorial idea and can be taught without a GPU, external data, or robot hardware.

**Independent Test**: Run the local notebook on CPU. It must explain the inputs and limits of each path and label any toy calculation as an analogy.

**Acceptance Scenarios**:

1. **Given** the same model weights, **When** a learner follows the three inference paths, **Then** the notebook distinguishes goal-conditioned CEM from goal-blind BC and video–inverse inference.
2. **Given** a local toy or diagram, **When** it illustrates the method, **Then** it is labelled as a teaching analogue and is not reported as a Qantara measurement.

### User Story 2 - Check the released-checkpoint finding (Priority: P2)

As a researcher, I want to evaluate the three official inference paths on the paper's Push-T task so that I can check whether the reported trade-off appears with released checkpoints.

**Why this priority**: This is the bounded paper-linked validation path. It does not require training the models, but it does require the official data and a working renderer.

**Independent Test**: A reviewer can inspect pinned code and asset metadata, every planned checkpoint and evaluation case, raw episode outputs, and paper and local values in separate fields.

**Acceptance Scenarios**:

1. **Given** a qualified runtime and matched data, **When** all three Qantara checkpoints are evaluated with CEM, BC, and video–inverse settings, **Then** the report includes 50 episodes across five declared evaluation seeds for each of three training checkpoints and all three paths.
2. **Given** the exact paper evaluation-seed identities are unavailable, **When** a local fixed seed set is used, **Then** the report names it and calls the result a protocol replication, not an exact sample-level reproduction.
3. **Given** data identity, serialization trust, storage, or renderer gates fail, **When** preflight ends, **Then** no data or checkpoint transfer or long evaluation starts and the report records the failed gate.

### Edge Cases

- The paper uses the DINO-WM Push-T corpus, while the Qantara README describes its data as LeRobot / D4RL Push-T expert data. The required file is named `pusht_expert_train`; do not assume these sources or conversions are identical without a provenance check.
- The paper reports five evaluation seeds but does not name them. A local seed set can match the count and protocol but not the exact sample list.
- The official checkpoint files are complete serialized Python model objects. Loading them needs trusted source review and a controlled environment.
- The host's MuJoCo EGL smoke failed, and its OSMesa backend did not import. A passing CUDA check alone does not qualify the evaluator.
- One checkpoint or one evaluation seed is incomplete evidence for the paper's three-checkpoint, five-evaluation-seed estimate.

## Requirements

### Functional Requirements

- **FR-001**: The notebook MUST explain Qantara's bridge-flow objective and its three inference paths using terms consistent with the current paper version.
- **FR-002**: The core notebook MUST run on the project's CPU environment without external data, model weights, new dependencies, or robot hardware.
- **FR-003**: The notebook MUST label each toy or simplified calculation as an analogy, not a Qantara implementation or paper reproduction.
- **FR-004**: The official evaluation target MUST be Table 2's Push-T result for the same Qantara checkpoint queried with CEM, BC, and video–inverse inference; it MUST use training checkpoint seeds 11, 22, and 33, 50 episodes, five declared evaluation seeds, and the paper's 50-step episode budget.
- **FR-005**: The evaluation MUST use the official inference settings: CEM with K=4, BC with K=2, and video–inverse with K=2. It MUST preserve the paper's goal-conditioned versus goal-blind distinction.
- **FR-006**: Before transfer, the run record MUST identify each artifact's source revision, size, advertised hash, license or terms, and local content hash after transfer. It MUST record the exact data-to-HDF5 provenance and any mismatch.
- **FR-007**: The evaluation MUST NOT start until the data identity, checkpoint serialization, pinned runtime, storage, and headless rendering gates pass.
- **FR-008**: The run record MUST preserve all per-episode outcomes, checkpoint seed, evaluation seed, method, runtime, memory, software, hardware, failures, and deviations. Raw records MUST remain outside Git.
- **FR-009**: The report MUST keep paper values separate from local values and report population standard deviation across training-seed means; it MUST label any missing checkpoint, episode, or evaluation seed as partial.
- **FR-010**: README, curriculum coverage, roadmap, reproduction matrix, project alignment, remaining work, and project tasks MUST agree on the tutorial and reproduction status.

### Key Entities

- **Inference path**: One checkpoint query mode: goal-conditioned CEM, goal-blind BC, or goal-blind video–inverse.
- **Checkpoint record**: One official Qantara training seed, source revision, file size, advertised and local hash, and load status.
- **Evaluation case**: One selected start and future goal from an episode, keyed by the declared evaluation seed and source episode/step.
- **Evaluation run**: One checkpoint, inference path, evaluation seed, fixed case set, outcome rows, and compute record.
- **Reproduction report**: Paper target, local measurements, raw-record location, deviations, missing inputs, and claim status.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every notebook cell runs on CPU from the project environment without external assets or new dependencies.
- **SC-002**: The notebook distinguishes three inference paths and states which require a goal observation.
- **SC-003**: A full evaluation contains 750 episodes per inference path and 2,250 episode records total, with no failed or missing case omitted.
- **SC-004**: The report shows the three paper values beside local results and calculates population standard deviation across the three training-seed means.
- **SC-005**: If any gate fails, no Push-T dataset or checkpoint is transferred, and the local status remains Not run or partial.

## Assumptions

- The core tutorial is local and CPU-sized. The official checkpoint evaluation is optional and requires a compatible NVIDIA runtime with working headless rendering.
- The paper's reported values are Qantara CEM 90.1 ± 1.1, BC 82.1 ± 1.0, and video–inverse 83.2 ± 0.6 success rate on Push-T. These are paper values, not local results.
- The exact five evaluation-seed identities are not stated in the paper or current official source. A declared local set therefore cannot be called an exact sample-level reproduction.
- No model training is in scope. The evaluation uses the three released Qantara checkpoints.
- No external file is committed to the repository.
