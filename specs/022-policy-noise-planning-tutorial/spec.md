# Feature Specification: Policy-Noise Planning Tutorial

**Feature Branch**: `[022-policy-noise-planning-tutorial]`
**Created**: 2026-10-15
**Status**: Draft
**Input**: Project goal to deliver roadmap lessons with reproducible local evidence; roadmap candidate for LeWAM diffusion-steering MPC.

## User Scenarios & Testing

### User Story 1 - Compare action-space and policy-noise planning (Priority: P1)

As a learner, I want to compare plans sampled as raw actions with plans sampled through a policy's noise input so I can see how a planner can exploit an inaccurate dynamics model.

**Why this priority**: The roadmap identifies policy-noise planning as a recent world-action-model idea, but its paper has no linked official code or checkpoint. A bounded local analogy can teach the mechanism without claiming to reproduce that work.

**Independent Test**: Run the notebook in a fresh CPU kernel. It must compare both planners on common task starts, use the same toy dynamics model and planning budget, report closed-loop outcomes and model-call counts, and label the example as illustrative.

**Acceptance Scenarios**:

1. **Given** a toy policy that maps noise to actions and a deliberately imperfect forward model, **when** the learner compares raw-action and policy-noise planning, **then** the notebook shows which candidate actions lie outside the policy's declared action support and how model predictions differ from the toy environment.
2. **Given** the same task starts and planning budget, **when** the notebook runs both planners and a direct-control reference, **then** it reports paired closed-loop success, steps, model calls, and variation across evaluation seeds.
3. **Given** the paper's reported result, **when** the notebook discusses it, **then** it links to the primary paper and keeps paper values separate from local measurements.
4. **Given** a clean project environment, **when** the notebook runs, **then** it needs no downloaded data, model weights, robot hardware, or added dependency.

## Requirements

### Functional Requirements

- **FR-001**: Add a notebook that explains the paper's policy-noise planning idea: optimize the noise input to a generative policy, decode actions at imagined states, score predicted goal distance, and execute the first action before replanning.
- **FR-002**: The local example MUST use one explicit toy policy, one imperfect toy forward model, and one ground-truth toy environment. It MUST identify that the model error and policy support are hand-authored.
- **FR-003**: Compare raw-action sampling, policy-noise sampling, and a direct-control reference on the same fixed task definition and paired starts.
- **FR-004**: Give the two planners the same candidate count, planning horizon, and replanning limit. Report these settings, the evaluation seeds, episode outcomes, success, steps, model calls, and variation across evaluation seeds.
- **FR-005**: Report prediction error and whether selected raw actions lie outside the toy policy's support. Do not present the constructed model error as evidence that real world models have the same failure.
- **FR-006**: Label the lesson as an illustrative analogy, not LeWAM code, a JEPA implementation, or a reproduction of the paper's benchmark.
- **FR-007**: Align the roadmap, curriculum coverage, reproduction matrix, README, project research, project task ledger, alignment record, and remaining-work record with the completed local lesson and the separate paper-reproduction gate.

### Out of Scope

- Implementing or training LeWAM, a flow-matching policy, or a learned world model.
- Downloading paper data, checkpoints, or external code.
- Claiming to reproduce arXiv:2610.12407 or validating its reported planning advantage.
- Adding a project dependency, planner library, or general-purpose evaluation framework.

## Success Criteria

- **SC-001**: Every code cell completes in a fresh CPU kernel with no external data or model assets.
- **SC-002**: The notebook reports paired closed-loop outcomes for at least 20 common starts across five declared evaluation seeds, with training-run variation explicitly marked not applicable because the toy models are not trained.
- **SC-003**: Both planners use the same candidate count, horizon, and episode limit; the notebook reports actual forward-model calls and success for all methods.
- **SC-004**: The notebook demonstrates the chosen toy model's off-support prediction error and states at least three limits of the analogy.
- **SC-005**: Every project index that lists current lesson coverage identifies Notebook 20 as illustrative and keeps the official paper evaluation separate.

## Assumptions

- The primary research source is the v1 paper, [LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC](https://arxiv.org/abs/2610.12407), submitted 2026-10-08.
- The paper's noise-space search and policy-support projection are suitable for a small hand-authored lesson, provided the local result is not described as a paper reproduction.
- The project uses standard-library Python for this small CPU lesson. The existing notebook environment is sufficient.
- The paper-scale evaluation stays a separate candidate until official code, checkpoint, benchmark data, terms, and an affordable host are verified.
