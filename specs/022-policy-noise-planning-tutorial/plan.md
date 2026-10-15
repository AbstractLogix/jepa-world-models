# Implementation Plan: Policy-Noise Planning Tutorial

**Branch**: `022-policy-noise-planning-tutorial` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a CPU-only notebook that demonstrates why sampling a generative policy's noise can constrain a planner to its action support. Compare this method with raw-action sampling under the same hand-authored, imperfect model and with a direct-control reference. Treat all measured results as an illustrative toy outcome.

## Technical Context

**Language/Version**: Python 3.10–3.13.
**Primary Dependencies**: Python standard library only.
**Storage**: Notebook and small text evidence report in Git; no downloaded data, model weights, or raw episode files in Git.
**Testing**: Execute every notebook code cell in a fresh CPU kernel; verify paired cases, equal planner budgets, result summaries, and claim labels.
**Target Platform**: Jupyter on CPU.
**Project Type**: Tutorial notebook and evidence documentation.
**Performance Goals**: Complete the local evaluation in under one minute on a standard CPU.
**Constraints**: No training, external model, robot hardware, or new dependency. Use fixed evaluation seeds and starts. The policy and dynamics model are hand-authored. The example is not a paper reproduction.
**Scale/Scope**: One notebook, one run report, and updates to the project curriculum and task records.

## Constitution Check

- Teach from evidence: PASS. Explain the selected paper idea and cite the primary source. Keep paper claims separate from local results.
- Label teaching examples honestly: PASS. Name the notebook as an illustrative analogy and state the synthetic setup.
- Make the learning path usable: PASS. Use the standard library and CPU only.
- Measure before performance claims: PASS. Compare paired starts and report outcomes, model calls, evaluation-seed variation, and the direct-controller baseline.
- Keep changes small and reviewable: PASS. Add one notebook, one report, and index/spec alignment. Do not add a framework or dependency.

## Design Decisions

- Use a one-dimensional goal task with two action components. The toy policy emits actions on a hand-authored behavior manifold; the toy forward model adds a false progress term off that manifold; the toy environment ignores that term.
- At each decision, sample the same number of candidate sequences for raw-action and policy-noise planners. Use the same horizon, predicted-goal objective, action penalty, and closed-loop step limit.
- Use random shooting (best-of-N) as the planner. The paper describes a sampling-based planner and states that its first diffusion-steering iteration is best-of-N; the local code does not implement its CEM/MPPI/CMA-ES updates or typical-set projection.
- Compare a simple state-feedback controller as a reference. Report that it observes the toy state directly and uses no model calls.
- Report variability across five evaluation-seed groups. Do not call these training seeds or model uncertainty.
- Keep raw per-episode results out of Git. The notebook regenerates the run; the report stores only aggregate results and run conditions.

## Project Structure

```text
specs/022-policy-noise-planning-tutorial/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/requirements.md
└── tasks.md
notebooks/20_policy_noise_planning.ipynb
docs/reproductions/policy-noise-planning.md
```

**Structure Decision**: Keep the example in one notebook and its summary in one report. Reuse the shared project evaluation protocol. Do not add source modules or general-purpose tooling.
