# Feature Specification: Matched Visual JEPA Decision Study

**Feature Branch**: `codex/matched-visual-jepa-decisions`
**Created**: 2026-10-15
**Status**: Implemented locally; evidence matrix complete; scientific comparison partial
**Input**: User request to design and run a controlled follow-up to the merged Notebook 22 study.

## User Scenarios & Testing

### User Story 1 - Validate the planner before comparing models (Priority: P1)

A researcher wants to know whether the planner can solve the task when it uses exact dynamics. This prevents planner failures from being mistaken for learned-model failures.

**Why this priority**: Notebook 22's exact-dynamics CEM reached 0/20 goals. Its short horizon and endpoint score do not establish whether the dynamics or the planner caused that result.

**Independent Test**: Run the declared planner search on development starts. The report records the tested settings, success, task return, steps, calls, and time. It locks one setting before any learned-model result is collected.

**Acceptance Scenarios**:

1. **Given** the fixed development starts and bounded search grid, **When** exact-dynamics CEM runs, **Then** the complete search results are saved without reading final evaluation outcomes.
2. **Given** the selected exact-dynamics planner, **When** it succeeds on fewer than 18 of 20 development starts, **Then** the held-out learned-model evaluation remains closed and the report states that the planner gate failed.
3. **Given** a qualifying development result, **When** the planner is locked, **Then** its settings and code identity are recorded before the final evaluation starts.

### User Story 2 - Compare model and prior value on shared starts (Priority: P1)

A researcher wants to measure whether an action-conditioned visual JEPA adds decision value beyond direct control, planner initialization, and other dynamics models.

**Why this priority**: The historical 100/100 prior-guided JEPA result and 20/20 direct-controller result both use the same hand-written momentum rule. A matched comparison must isolate the model contribution.

**Independent Test**: When the planner gate passes, the locked v1 matrix is evaluated on the same held-out starts, task, outcome rules, and planner budget. State-only methods and any omitted primary baselines are labeled, and the claim boundary matches those limits.

**Acceptance Scenarios**:

1. **Given** one locked planner protocol, **When** a model is evaluated, **Then** uniform and policy-prior initialization use the same horizon, candidates, elites, updates, action limits, and replanning rule.
2. **Given** visual predictions from different methods, **When** a common action ranking is reported, **Then** each method uses the same task-grounded outcome and any visual state probe was fit without held-out outcomes.
3. **Given** a missing, failed, or collapsed model run, **When** results are summarized, **Then** its cases remain in the records and the comparison is labeled partial.
4. **Given** five trained model seeds and shared evaluation starts, **When** uncertainty is reported, **Then** training-seed variation is kept separate from evaluation-start variation.
5. **Given** a prior-only planner or learned direct-policy baseline is absent, **When** the report states its conclusion, **Then** the omission is visible and incremental JEPA value remains unresolved.

### User Story 3 - Reproduce and inspect the study (Priority: P2)

A contributor wants a bounded CPU smoke path and a documented CUDA path that can regenerate the records and figures.

**Why this priority**: The study must be inspectable without downloading a dataset, checkpoint, or robot simulator.

**Independent Test**: A clean CPU environment runs unit checks and the smoke protocol. A CUDA environment can run the gated study with outputs directed outside the repository to E: or D:.

**Acceptance Scenarios**:

1. **Given** only the locked project environment, **When** a contributor runs smoke mode, **Then** it uses reserved smoke seeds, writes schema-valid records, and does not open final cases.
2. **Given** retained CSV evidence, **When** the figure command runs, **Then** two concise charts are generated deterministically without training or external data.
3. **Given** a report claim, **When** a reader checks its evidence links, **Then** the source record, sample count, method access, and evidence status agree.

## Edge Cases

- The exact-dynamics planner may fail its development gate. Preserve its records and do not open learned-model evaluation cases.
- The final oracle gate may fail on held-out starts even after it passed development. Preserve oracle rows, stop the learned-model stage, and do not interpret later model failures as model failures.
- A predicted visual embedding may be finite but lose position, velocity, or action-sensitive information. Report probe and intervention diagnostics separately from planning outcomes.
- A model may collapse, return non-finite values, or fail to load. Keep its run row and mark the comparison partial.
- CEM may use the full episode budget without reaching the goal. Keep the episode in all denominators.
- A direct controller may use no model calls and less elapsed time. Report measured compute and do not call the result compute-matched.
- GPU initialization or synchronization may fail. Use the CPU smoke path and mark any incomplete CUDA run.

## Requirements

### Functional Requirements

- **FR-001**: The study MUST preserve Notebook 22 and its evidence as a historical synthetic MountainCar result. New code, data, figures, and reports MUST use a new versioned name.
- **FR-002**: The methodology audit MUST identify the historical planner horizon, candidate count, elite count, update count, initial distribution, rollout score, replan frequency, model access, seeds, splits, and score units. It MUST separate observed facts from interpretation.
- **FR-003**: Before tuning, the study MUST record the planner search grid, development seeds, compute ceiling, success rule, selection rule, and stop rules. It MUST use development outcomes only to select planner settings.
- **FR-004**: Exact-dynamics planner validation MUST use a separate development set. The locked planner MUST reach at least 18/20 development starts to pass the development gate. If no tested setting passes, final learned-model evaluation MUST remain closed.
- **FR-005**: Before learned-model evaluation, the locked exact-dynamics planner MUST pass a 18/20 held-out oracle gate on the declared test starts. If it fails, retain oracle results and do not interpret learned-model failures.
- **FR-006**: When the gates pass, the comparison MUST include exact-dynamics planning, learned state-space dynamics, learned visual reconstruction dynamics, action-conditioned visual JEPA dynamics, direct visual-momentum control, uniform and momentum-prior search for each supported dynamics model, and a prior-only planning baseline. If a condition is absent from the frozen v1 matrix, the report MUST mark it not run and MUST leave the related incremental-value claim unresolved.
- **FR-007**: Methods MUST share training data where applicable, evaluation starts, task, episode limit, success rule, action bounds, and declared planner budget. State-only methods MUST be labeled privileged. Direct control MUST have one outcome per evaluation start, not one duplicated outcome per model seed.
- **FR-008**: The comparison MUST report success, episode return, steps to goal, actual elapsed time, planning time, transition-model calls, action-ranking agreement against a common task-grounded outcome, prediction/intervention fidelity, and representation diagnostics. Model-native losses MUST remain labeled in their own units.
- **FR-009**: Position and velocity probes for visual representations MUST be fit from training data, selected only with validation data, and evaluated on separate test data. Probe scores MUST be labeled diagnostic, not proof of planning value.
- **FR-010**: The preregistration MUST state the hypothesis, primary comparison and outcome, task, data splits, training and evaluation seeds, planner budgets, compute ceiling, failure and abstention criteria, and planned uncertainty analysis before final evaluation.
- **FR-011**: The study MUST keep model-training seeds separate from evaluation-start seeds and report positive, negative, partial, and inconclusive outcomes.
- **FR-012**: The study MUST preserve versioned machine-readable evidence, per-run failures, software/device metadata, and evidence-validation results. It MUST NOT overwrite Notebook 22 records.
- **FR-013**: The study MUST provide a CPU smoke path and optional CUDA path. It MUST add no package, external dataset, checkpoint, or robot-hardware requirement. Large outputs MUST be directed to E: or D: when available.
- **FR-014**: The study MUST provide two deterministic, accessible figures from retained evidence. The first figure MUST show (A) the historical Notebook 22 and new v1 planner outcomes as separate protocols, including an explicit not-run marker for prior-only search; (B) per-seed action-ranking agreement against a common task outcome; (C) embedding spread, selected probes, and predicted action effects; and (D) closed-loop success against model-transition calls and measured time. The second figure MUST show the experiment architecture. Captions MUST state the task, metrics, sample counts, access differences, evidence version, and claim limits.
- **FR-015**: The research report, reproduction matrix, roadmap, curriculum coverage, case study, and portfolio manifest MUST use the same evidence status and claim limits.
- **FR-016**: The report MUST cite primary sources and MUST NOT claim a paper reproduction, a general JEPA advantage, production robotics ability, statistical significance without an appropriate analysis, or independent replication.

### Key Entities

- **Planner protocol**: Locked horizon, candidate budget, elite count, update count, action representation, initialization, rollout score, replanning interval, and code hash.
- **Model run**: Model family, model seed, training-data seed, split, objective, probe identity, device, software, runtime, memory, metrics, and status.
- **Intervention case**: Held-out visual context, paired candidate actions, predicted and actual next state, common task outcome, model seed, and status.
- **Evaluation episode**: Method, optional model seed, shared start seed, outcome, calls, timing, device, and failure reason.
- **Experiment figure**: Source evidence version, transformation, sample count, uncertainty method, and known limits.

## Success Criteria

### Measurable Outcomes

- **SC-001**: The audit report lists the historical Notebook 22 protocol facts and at least five methodology risks or limits with links to their source records.
- **SC-002**: Planner development either locks an exact-dynamics setting that succeeds on at least 18/20 development starts, or leaves final learned-model cases unopened and records the failed gate.
- **SC-003**: The held-out oracle gate either succeeds on at least 18/20 starts and unlocks the comparison, or closes the learned-model evaluation with no unsupported model-failure interpretation.
- **SC-004**: A completed comparison contains at least five training seeds per trained method and at least 20 shared evaluation starts. Any lower count is labeled exploratory or partial with missing cases listed.
- **SC-005**: Every completed method has versioned records for outcomes, prediction/intervention fidelity, model calls, actual time, access level, seeds, device, and failures.
- **SC-006**: The common action-ranking score is based on the same task outcome for every model; native latent, pixel, and state losses are not pooled.
- **SC-007**: The A–D evidence dashboard and E architecture figure regenerate from retained evidence and the frozen protocol with deterministic commands. The generator fails closed when expected rows are missing and reports the prior-only arm as not run.
- **SC-008**: CPU unit tests and smoke checks pass without opening final evaluation cases. CUDA tests or experiments are reported separately from CPU checks.
- **SC-009**: All public project records link to the new report and describe it as complete, partial, gated, or exploratory consistently.
- **SC-010**: The report identifies the missing prior-only planner and learned direct-policy baselines. It makes no claim that the visual JEPA adds decision value without those comparisons.

## Assumptions

- The study continues to use the synthetic `MountainCarContinuous-v0` task and Notebook 22 renderer.
- The existing direct visual-momentum controller is a scripted policy baseline, not a learned policy.
- The visual reconstruction model and action-conditioned JEPA remain small local analogues. Neither is a paper reproduction.
- Exact and state-space methods can read simulator state and are privileged diagnostics.
- Training uses new fixed data and model seeds. Development, intervention, smoke, and final evaluation seeds do not overlap.
- The full comparison uses five trained model seeds unless the locked compute ceiling prevents completion. The evidence threshold is not lowered after results are seen.
- CUDA is optional. The project CPU environment remains the contributor baseline; the prepared E:-hosted CUDA environment may run the full study.
- The final test set is opened first for the predeclared oracle gate only. Learned-model results are collected only if that gate passes.

## Research Basis

- [V-JEPA 2](https://arxiv.org/abs/2506.09985) motivates action-conditioned prediction in visual latent space. This study does not reproduce its architecture or robot result.
- [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/abs/2512.24497) studies how model, objective, and planning choices affect decision outcomes. Its authors tune planning settings before fixing the planner for model ablations. This small synthetic study uses that as a methodology reference, not a result to reproduce.
- [Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization](https://proceedings.mlr.press/v120/bharadhwaj20a.html) describes action-sequence search and its computational trade-offs.
- [Gymnasium MountainCarContinuous](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) defines the task transition, action cost, goal reward, and start-state distribution.
- The [project evaluation protocol](../../docs/EVALUATION_PROTOCOL.md) separates model prediction, action intervention, closed-loop outcome, decision benefit, and compute evidence.
