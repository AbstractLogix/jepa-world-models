# Implementation Plan: LeHami Symplectic Dynamics Tutorial

**Branch**: `014-lehami-symplectic-tutorial` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/014-lehami-symplectic-tutorial/spec.md`

## Summary

Add a CPU-only notebook that explains LeHami's symplectic structure with a small harmonic-oscillator calculation. Show determinant and polygon-area preservation under leapfrog, and contrast it with explicit Euler. Keep the numerical demonstration separate from the published Table 1 values and the local one-seed-per-model probe values. Update the project map and roadmap to describe the lesson and keep the four pending runs open.

## Technical Context

**Language/Version**: Python 3.10–3.13  
**Primary Dependencies**: Python standard library for the calculation; existing Jupyter environment to open the notebook  
**Storage**: None; use in-memory numeric arrays  
**Testing**: Execute the notebook from start to finish on CPU; no separate test suite  
**Target Platform**: Local notebook environment on CPU  
**Project Type**: Educational notebooks and documentation  
**Performance Goals**: Complete the numeric demonstration in seconds on a CPU  
**Constraints**: No downloads, checkpoints, robot hardware, or additional training runs. Do not depend on files under `/tmp`. Use Python's standard double-precision floats for the numeric preservation check. Keep the paper's action-held-within-a-step assumption and all evidence limits clear.  
**Scale/Scope**: One new tutorial notebook, one local execution record, and alignment changes to existing curriculum and roadmap documents.

## Constitution Check

- Teach from evidence: Pass. Link the primary paper and official source. Label the paper values and local measurements separately.
- Label teaching examples honestly: Pass. The harmonic oscillator is a mathematical analogue, not a trained model or paper reproduction.
- Make the learning path usable: Pass. The numeric example uses existing local dependencies and no external data or hardware.
- Measure before performance claims: Pass. Report only the specified determinant, area, and recorded K=0 probe measurements.
- Keep changes small and reviewable: Pass. No dependency, shared source code, or model training changes are planned.

## Research Decisions

See [research.md](research.md). The key decision is to separate a deterministic structure demo from the stored paper and local probe measurements. The paper says volume preservation does not bound rollout error, and symplecticity does not establish that a latent model represents physical dynamics.

## Design

### Notebook flow

1. State the learning outcomes, prerequisites, evidence type, and non-goals.
2. Define a phase-space point and the symplectic matrix condition in direct language.
3. Build the leapfrog matrix for `H(q,p) = (q² + p²)/2`; show its determinant and polygon-area ratio for several step sizes.
4. Compare explicit Euler on the same oscillator and label it only as a numerical contrast.
5. Explain the LeHami action-conditioned state and the held-action condition. State that time-varying actions can change energy while each step remains symplectic.
6. Present the paper's Table 1 results beside the local seed-0 values, with run counts, probe, and missing runs.
7. Summarize what the guarantee and the measured K=0 probe do and do not support.

### Data and evidence handling

- Compute the toy example from constants in the notebook with the Python standard library. Do not import the research checkout or require generated pendulum data.
- Keep paper values, local values, and toy calculations in separate labels and sections.
- Link local measurements to the reproduction report. Do not write a local aggregate from one seed per model.
- Keep temporary checkpoints and datasets outside the repository.

### Project structure

```text
notebooks/15_lehami_symplectic_dynamics.ipynb
docs/evidence/lehami-symplectic-tutorial.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
README.md
specs/001-project-vision/tasks.md
specs/001-project-vision/alignment.md
specs/001-project-vision/remaining-work.md
specs/014-lehami-symplectic-tutorial/
```

The notebook is the only learner-facing implementation. No external interface or new shared code is needed, so no contract file is required.

## Validation Strategy

- Execute all notebook code cells in order in a clean Python namespace. A Jupyter front end was not installed on the current host, so the front-end path remains unverified here.
- Confirm that every code cell completes without data or model downloads.
- Confirm the absolute determinant and polygon-area errors are at most `1e-12` in float64 for the displayed leapfrog step sizes.
- Confirm the explicit Euler determinant differs from one and that the notebook does not equate Euler with LeWM.
- Review the displayed paper and local probe labels and all cross-links.
- Run Spec Kit analysis and convergence after implementation, then check the roadmap, curriculum map, and project tasks against the completed notebook and evidence record.

## Complexity Tracking

No constitution violations or added complexity.
