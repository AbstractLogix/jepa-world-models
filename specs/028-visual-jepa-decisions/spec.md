# Feature Specification: Learned Visual JEPA Decision-Usefulness Lesson

**Feature Branch**: `028-visual-jepa-decisions`
**Created**: 2026-10-15
**Status**: Draft
**Input**: Continuing project goal to compare action-conditioned prediction, planning, and direct control with reproducible local evidence.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Inspect learned visual predictions (Priority: P1)

A learner wants to compare an action-conditioned JEPA with other dynamics models when the observation is an image.

**Why this priority**: The project has action-conditioned planning analogues, but it does not yet evaluate a trained visual JEPA. This lesson fills that central evidence gap.

**Independent Test**: A fresh project kernel on CPU or a supported CUDA device produces held-out action-intervention records for each model, keeps all actions from one starting state together, and reports prediction and ranking measures.

**Acceptance Scenarios**:

1. **Given** a held-out state and at least three candidate actions, **When** each model predicts the next outcome, **Then** the record includes the predicted and observed outcome for every action from that same state.
2. **Given** a visual latent model and an action-conditioned JEPA, **When** the learner compares them, **Then** the lesson states their different training objectives and reports each prediction metric in its own units.
3. **Given** a collapsed, non-finite, or failed model run, **When** the lesson summarizes results, **Then** it preserves the failure and labels the comparison partial.

### User Story 2 - Compare decisions on paired starts (Priority: P1)

A learner wants to know whether an accurate predictive model improves action selection and closed-loop outcomes.

**Why this priority**: Prediction quality alone does not show whether a model helps a planner choose actions.

**Independent Test**: The learner can inspect paired outcomes for oracle dynamics, learned state-space dynamics, learned visual latent dynamics, action-conditioned JEPA planning, a direct controller, and policy-guided JEPA planning on the same evaluation starts.

**Acceptance Scenarios**:

1. **Given** a fixed set of evaluation starts, **When** each method is evaluated, **Then** it uses the same task, start states, episode limit, and stated planning budget where applicable.
2. **Given** the completed episodes, **When** the learner reads the report, **Then** it shows every outcome, per-method success and return, paired differences, model calls, elapsed time, and missing runs.
3. **Given** state-based oracle or learned-model methods and image-based methods, **When** the results are compared, **Then** the report identifies privileged state access and does not present the methods as an equal-observation benchmark.
4. **Given** unconstrained JEPA planning and policy-guided JEPA planning, **When** the learner compares them, **Then** the report keeps their candidate generation separate and applies the same candidate and rollout limits.

### User Story 3 - Repeat the local study (Priority: P2)

A contributor wants to rerun the lesson without private data, robot hardware, or a large model download.

**Why this priority**: The lesson must be usable as part of the project's CPU learning path.

**Independent Test**: A contributor follows the project setup guide, runs the notebook in a fresh CPU kernel or an optional supported CUDA kernel, and regenerates its episode and intervention records.

**Acceptance Scenarios**:

1. **Given** the locked project environment, **When** the contributor runs the lesson, **Then** it needs no external dataset, checkpoint, or robot hardware.
2. **Given** the saved records and report, **When** a contributor checks the stated evidence, **Then** the method labels, episode counts, pairing, seeds, and claim limits agree across the notebook and project documentation.

### Edge Cases

- A dynamics model may return a non-finite value or an invalid state. Preserve the failure in its source record.
- A model may collapse to constant embeddings. Report the collapse diagnostic and do not count the model as a successful JEPA run.
- An episode may time out or fail before the goal. Keep it in the denominator.
- A direct controller may use less compute than a model-based planner. Report the measured difference and do not claim a matched-compute advantage.
- Rendering or simulator setup may fail. Record the environment failure separately from a low task score.
- If a planned training seed or evaluation episode is missing, label the result partial and state why.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The lesson MUST identify its results as a local, small-scale MountainCar study. It MUST NOT call the study a reproduction of V-JEPA 2, LeJEPA, or another paper benchmark.
- **FR-002**: Before final evaluation, the lesson MUST define the task, renderer, trajectory collection, train and validation splits, held-out intervention states, evaluation starts, success rule, episode limit, seeds, and planning budget.
- **FR-003**: The model training data MUST pair two-frame rendered observation contexts with actions and observed next-frame contexts. Evaluation states MUST NOT be used for training or model selection.
- **FR-004**: The lesson MUST preserve at least 60 held-out intervention outcomes from at least 20 starting states, with at least three actions per state. It MUST label each method's prediction and endpoint-scoring space; latent or pixel score agreement MUST NOT be described as task-reward accuracy.
- **FR-005**: The comparison MUST include oracle dynamics, a learned state-space model, an image-reconstructing latent dynamics model, an action-conditioned JEPA, a direct visual controller, and policy-guided planning with the JEPA.
- **FR-006**: The action-conditioned JEPA MUST predict a target visual representation from a two-frame observation and action. It MUST use a non-collapse objective or diagnostic and MUST report whether its representation collapsed. The screen flags non-finite embeddings or more than half of embedding dimensions with standard deviation below `1e-3` on the fixed validation set. A validation-only pilot with a matching training-protocol hash, objective settings, split, seeds, and device MUST pass before final model training, held-out intervention collection, or episode evaluation starts. Smoke cases MUST use separate seeds and MUST NOT be included in the final comparison. If the pilot fails or is missing, full mode MUST skip final training and evaluation, keep the comparison partial, and leave final cases unopened.
- **FR-007**: Each method MUST be evaluated on at least 20 paired closed-loop starts. Trained methods MUST use at least five independent training seeds for a comparison result; a smaller run MUST be labeled exploratory or partial.
- **FR-008**: The report MUST include per-seed and per-episode records, success, return, paired differences, prediction or intervention measures, model calls, device, software versions, elapsed time, peak GPU memory when available, and any failed or missing runs.
- **FR-009**: The report MUST keep training variation separate from evaluation-case variation, identify privileged state access, and avoid claims of advantage when compute or observation budgets differ.
- **FR-010**: The lesson MUST use the project evaluation protocol and MUST add no dependency, downloaded dataset, external checkpoint, or robot-hardware requirement. It MUST select CUDA when an installed PyTorch build supports the visible GPU, and otherwise run on CPU. GPU use MUST remain optional.
- **FR-011**: The README, curriculum coverage, research roadmap, reproduction matrix, project alignment, remaining-work inventory, and project task ledger MUST describe the lesson and its evidence boundary consistently.

### Key Entities *(include if feature involves data)*

- **Training transition**: A two-frame rendered observation context, action, next two-frame context, next frame, and simulator state used only by the declared training or validation split.
- **Intervention case**: One held-out start evaluated under several actions, with the observed outcome, each method's prediction, and outcome ranking.
- **Evaluation episode**: One method run from a fixed start, with success, return, steps, elapsed time, compute count, and completion status.
- **Model run**: One trained model with its training seed, data identity, objective, collapse diagnostic, and measured results.
- **Planner condition**: A model and candidate-generation setting, including whether candidates are unconstrained or policy-guided.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The lesson records at least 60 action interventions from 20 or more held-out starts, with three or more actions per start.
- **SC-002**: The lesson records at least 20 paired closed-loop outcomes for every method and every planned training seed, or labels the comparison partial and lists each missing run.
- **SC-003**: The trained-model comparison contains at least five independent training seeds per trained method; it reports seed-level results separately from variation across evaluation starts.
- **SC-004**: The report gives prediction diagnostics, task success and return, paired differences, compute counts, elapsed time, and observation and compute differences.
- **SC-005**: A fresh CPU project environment completes the lesson without downloading a dataset or checkpoint. A supported CUDA setup uses the GPU for model training and planning. Every notebook and report claim is limited to the stated local task and protocol.
- **SC-006**: All seven project records named in FR-011 link to the same lesson and state whether its evidence is smoke, exploratory, comparison, or incomplete.

## Assumptions

- The lesson uses the existing MountainCarContinuous task and deterministic frame generator from Notebook 16.
- The direct baseline is the existing visual momentum controller. It is a scripted controller, not a learned policy.
- The policy-guided JEPA condition uses a scripted visual-momentum action prior to seed CEM candidates. It is not a learned policy. The unconstrained JEPA condition samples from the uniform initial CEM distribution.
- Oracle and learned state-space methods can read simulator state. They are diagnostic references with privileged state access.
- The visual latent baseline reconstructs the next frame and predicts future latents. The JEPA predicts target latents from a two-frame context without an image decoder.
- The action-conditioned JEPA is a compact model trained from this lesson's generated trajectories. Its use of a JEPA objective does not make this a V-JEPA 2 implementation or paper reproduction.
- The project's evaluation protocol sets five independent training seeds as the default for a comparison. A measured compute limit may leave the work partial; it does not lower the evidence threshold.
- CPU remains the supported base path. On a compatible GPU host, the tutorial selects CUDA for model training and batched planning, then records the selected device, precision, package versions, runtime, and peak GPU memory.
- Use the existing `requirements-torch.txt` CUDA 12.8 override for GPU setup. Keep the locked CPU wheels as the default and add no package dependency.
- No external assets are needed. Generate episode data on demand. Keep any larger outputs, model weights, and caches outside the repository on E: or D:, and commit only small, reproducible evidence records.

## Research Basis

- [V-JEPA 2](https://arxiv.org/abs/2506.09985) motivates action-conditioned prediction in visual latent space. This lesson does not reproduce its model or robot benchmarks.
- [LeJEPA](https://arxiv.org/abs/2511.08544) motivates a predictive objective with representation regularization. The lesson's compact setup is a teaching implementation, not the paper's scale or full training recipe.
- [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) studies dense and temporally consistent video features. [RoboJEPA](https://arxiv.org/abs/2610.10515) reports scaling trends for a large multi-embodiment world model. Both are research context only; this local study cannot validate their benchmark or scaling claims.
- The project's [evaluation protocol](../../docs/EVALUATION_PROTOCOL.md) and [decision-usefulness lesson](../../notebooks/14_decision_usefulness.ipynb) define the local evidence and comparison boundaries.
