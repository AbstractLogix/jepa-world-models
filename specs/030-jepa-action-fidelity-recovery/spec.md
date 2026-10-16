# Feature Specification: JEPA Action Fidelity and Representation Recovery

**Feature Branch**: codex/jepa-action-fidelity-recovery
**Created**: 2026-10-16
**Status**: Draft
**Input**: User request to diagnose the Notebook 23 JEPA representation and action-prediction failures with a preregistered, bounded study.

## User Scenarios & Testing

### User Story 1 - Explain the recorded v1 failure (Priority: P1)

A researcher wants to check the Notebook 23 claims against the saved records and identify which possible causes have evidence and which still need a test.

**Why this priority**: Model changes without an evidence audit can hide or mislabel the original failure.

**Independent Test**: Recompute the v1 counts and ranges from the retained CSV files. Confirm the analysis leaves all v1 files unchanged.

**Acceptance Scenarios**:

1. **Given** the retained v1 records, **When** the audit runs, **Then** it reports all five JEPA spread flags, probe scores, action effects, and planner outcomes with matching row counts.
2. **Given** a cause that was not measured in v1, **When** the report describes it, **Then** it labels the cause as a hypothesis and names the test that can confirm or reject it.

### User Story 2 - Test visual information and objective behavior (Priority: P1)

A researcher wants exact, privileged-state, handcrafted-visual, and supervised-visual controls before attributing a failure to a JEPA objective.

**Why this priority**: A visual model result is not interpretable if the observations or training protocol cannot support the target dynamics.

**Independent Test**: Run the fixed validation protocol on matched data and seeds, then confirm the positive controls and each objective’s validation records are complete.

**Acceptance Scenarios**:

1. **Given** the same rendered frames, actions, train/validation split, architecture size, and training budget, **When** the supervised visual control trains, **Then** the report separates it from privileged state and exact-simulator controls.
2. **Given** the three preregistered JEPA objectives, **When** training completes, **Then** every loss component, encoder and predictor gradient, embedding diagnostic, probe diagnostic, and action-effect diagnostic is retained per seed.

### User Story 3 - Open held-out tests only after validation (Priority: P1)

A researcher wants independent action-fidelity and multi-step evidence, with planning opened only if frozen checks pass.

**Why this priority**: One-step prediction loss and representation spread alone do not show that action-conditioned predictions support planning.

**Independent Test**: The runner rejects held-out and planning stages unless the locked protocol, validation records, positive control, and model gates pass.

**Acceptance Scenarios**:

1. **Given** any failed validation gate, **When** a held-out or planning stage is requested, **Then** the runner preserves current records and refuses to open the next stage.
2. **Given** a model that passes all held-out gates, **When** the bounded planning study runs, **Then** it uses the locked Notebook 23 planner budget and common starts and reports each seed and method separately.

## Edge Cases

- A non-finite embedding or loss invalidates that seed and remains in the records.
- A finite embedding may have low effective rank, weak velocity information, an unstable probe, or no action sensitivity.
- A decoded state may be clipped even when the unclipped readout is non-finite or outside the task bounds.
- An action effect can be zero in the exact simulator at a physical boundary; tied cases are excluded from action-order metrics and counted.
- A supervised visual control may fail. In that case, the study must investigate the observation and data protocol before blaming a JEPA objective.
- A JEPA candidate may pass representation checks and fail action-fidelity or multi-step checks. Planning remains closed.
- A run may stop at any phase. Missing rows and artifacts remain visible; no seed is replaced.

## Requirements

### Functional Requirements

- **FR-001**: The work MUST preserve Notebook 22, Notebook 23, their v1 protocols, and all existing v1 evidence bytes.
- **FR-002**: The audit MUST compute v1 claims from raw evidence and record any mismatch with existing prose.
- **FR-003**: The investigation report MUST separate confirmed findings from hypotheses and state the test for each hypothesis.
- **FR-004**: The v2 protocol MUST be versioned, hashed before v2 training, and state numeric collapse, probe, action-fidelity, rollout, compute, storage, and stop gates.
- **FR-005**: The diagnostic suite MUST record actual and predicted next states for actions -1, 0, and +1 at identical visual contexts, along with latent displacement, action sensitivity, clipping, and saturation.
- **FR-006**: The suite MUST report embedding variance, effective rank, encoder and predictor output norms, encoder and predictor gradient magnitudes, probe conditioning, coefficient magnitudes, and non-finite counts.
- **FR-007**: The controls MUST include exact simulator transitions, learned privileged state dynamics, a handcrafted visual state estimate, and a supervised visual dynamics model.
- **FR-008**: The objective comparison MUST use fixed SIGReg-style, VISReg-style, and variance/covariance settings on the same three model/data seed pairs and data splits. It MUST include the supervised visual control.
- **FR-009**: A failed supervised visual control MUST prevent a JEPA-objective failure claim until the visual observation or data protocol is investigated.
- **FR-010**: No held-out action-fidelity evaluation may open until the frozen validation gates pass. No planning evaluation may open until the selected model passes held-out action and multi-step gates.
- **FR-011**: The planning stage, if opened, MUST include exact dynamics, learned state dynamics, supervised visual dynamics, the validated JEPA, direct visual-momentum control, and JEPA-guided CEM with and without the same prior. The prior-only action baseline MUST not be counted as an independent learned planner.
- **FR-012**: Evidence records MUST retain every seed, failure, exclusion, sample count, device, runtime, memory, code hash, and protocol hash.
- **FR-013**: Figures MUST be deterministic, use retained measured data, show individual seed records, include sample counts and limits, and keep v1 and v2 evidence distinct.
- **FR-014**: Reports MUST distinguish learned JEPA models from state-space, supervised visual, and handcrafted controls. They MUST NOT claim a paper reproduction or general JEPA result from this synthetic task.
- **FR-015**: The final documentation MUST update the roadmap, evaluation protocol, reproduction matrix, technical report, portfolio case study, and portfolio manifest without changing the original v1 record.

## Key Entities

- **Protocol lock**: Version, thresholds, seed roles, candidate objectives, code hash, protocol hash, compute limit, and stage status.
- **Model run**: Objective, paired model/data seed, train/validation rows, loss components, gradient norms, embedding metrics, probe metrics, device, runtime, and status.
- **Action intervention**: One visual context paired with actions -1, 0, and +1, exact next states, predicted states and latents, clipping, and status.
- **Rollout record**: Shared test start, action sequence, horizon, predicted and actual states, latent distance, distribution distance, and method seed.
- **Planning episode**: Method, model seed, start seed, initialization, success, return, steps, transition calls, timing, device, and failure reason.
- **Figure record**: Figure ID, evidence version, source file hashes, sample count, uncertainty method, alt text, caption, and output hash.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The v1 audit reproduces 15 model rows, 10 probe rows, 960 intervention rows, and 660 episode rows without writing to them.
- **SC-002**: The v2 protocol is frozen and hashed before the first v2 model fit.
- **SC-003**: The validation run records all preregistered objective/control seed pairs or marks each missing run as failed or incomplete.
- **SC-004**: The report states whether the supervised visual control passes before making an objective-level conclusion.
- **SC-005**: Held-out data and planning records exist only if their prior-stage gates pass.
- **SC-006**: Every figure regenerates from versioned evidence and includes every completed seed.
- **SC-007**: The repository unit suite, Ruff, package build, evidence validation, and CPU smoke checks pass or record exact failures and unrun checks.

## Assumptions

- The study keeps the synthetic MountainCar renderer and task from Notebook 23.
- Existing v1 evidence is historical and may not be reclassified or rewritten.
- The validation study uses three paired model/data seeds and a fixed 35-epoch budget.
- CUDA may be used only for the bounded v2 fit and evaluation stages; all transient artifacts stay on E:.
- Planning is conditional. A negative validation or held-out result is a successful research outcome when the evidence is complete.
- The local SIGReg, VISReg-style, and variance/covariance implementations are objective adaptations, not full paper reproductions.
