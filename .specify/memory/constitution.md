# JEPA World Models Tutorial Constitution

## Core Principles

### I. Teach from evidence

Every claim about a paper, model, dataset, or result MUST link to a primary source. A paper's reported result MUST remain separate from a result measured in this repository.

### II. Label teaching examples honestly

Each lesson MUST say whether it is a concept demo, a simplified analogue, a small-scale experiment, or a paper reproduction. A tutorial MUST NOT imply that a toy environment, synthetic target, or reduced model reproduces a paper's result.

### III. Make the learning path usable

Each lesson MUST state its learning outcome, prerequisites, required data and tools, and expected output. The core path MUST have a local path that does not depend on robot hardware or private data. Optional downloads and GPU use MUST be clear before the learner starts.

### IV. Measure before making performance claims

Lessons that compare methods MUST name the baseline, task, metric, and run conditions. Reported measurements MUST include enough detail to repeat them. A target outcome or paper result MUST NOT be written as a result from this repository.

### V. Keep changes small and reviewable

Prefer the smallest change that improves learning or reliability. Keep existing notebook order and code layout unless evidence shows a clear problem. Add new tools, abstractions, and dependencies only when a lesson needs them.

## Educational and Evidence Constraints

- Use direct, simple English in learner-facing text. Keep technical terms when they are needed and define them at first use.
- The current project targets Python 3.10 through 3.13. Keep CPU PyTorch wheels in the locked `uv` environment; document CUDA wheel overrides separately from the base lock.
- State the source, license, size, and access limits for external data and model weights. Do not commit downloaded datasets, checkpoints, credentials, or generated outputs.
- Treat simulated control, synthetic data, and public robot data as different evidence. Do not imply physical robot validation from a simulation.
- Mark unknown runtime, hardware, data, and reproducibility claims as unverified until they are checked.

## Development Workflow

- Use Spec Kit for project-level changes and new tutorial modules. Write the intended learner outcome and acceptance criteria before implementation.
- Keep `spec.md`, `plan.md`, `tasks.md`, the curriculum coverage map, and the roadmap consistent with the current code.
- Use the cheapest effective check for each change. Do not run long training jobs or device-specific checks unless the change needs them; record checks that remain undone.
- Review source dates and links when updating the research roadmap. Give each research snapshot a review date.

## Governance

This constitution governs the project specifications and tutorial content. A change that conflicts with a principle MUST explain why in its plan and record the decision. Update this constitution when project goals or teaching rules change. Use semantic versioning: MAJOR for a change to an existing principle, MINOR for a new principle or section, and PATCH for wording that does not change meaning. Review the constitution, requirements checklist, and current code during convergence.

**Version**: 2.0.0 | **Ratified**: 2026-10-14 | **Last Amended**: 2026-10-15
