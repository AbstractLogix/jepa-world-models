# Implementation Plan: Sampling-Based Planning Tutorial

**Branch**: `023-sampling-based-planning-tutorial` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a CPU notebook that compares RRT, KPIECE, and CEM on a hand-authored point-mass detour task. The notebook will use one explicit transition rule, common evaluation starts, and an equal maximum transition-call budget per planning decision. All local results are illustrative state-space measurements. They do not test a learned model or reproduce the paper.

## Technical Context

**Language/Version**: Python 3.10–3.13.
**Primary Dependencies**: Existing project NumPy, Matplotlib, and Jupyter packages; no new dependency.
**Storage**: Notebook and short text report in Git. Do not commit downloaded assets or generated per-episode files.
**Testing**: Run all notebook cells in a fresh CPU kernel; validate paired starts, per-planning-call budget caps, episode counts, and claim labels.
**Target Platform**: Jupyter on CPU.
**Performance Goals**: Complete the 100-start local evaluation in under one minute on a standard CPU.
**Constraints**: No training, external model, data download, simulator, robot hardware, or additional dependency. Use fixed seeds and disclose that the state and transition rule are hand-authored.
**Scale/Scope**: One notebook, one run report, and alignment updates to the curriculum, evaluation protocol, roadmap, reproduction matrix, and project records.

## Constitution Check

- Teach from evidence: PASS. Cite the arXiv paper and the authors' lab publication listing. Keep paper claims separate from local measurements.
- Label teaching examples honestly: PASS. Call the notebook an illustrative state-space analogy and identify its exact toy dynamics.
- Make the learning path usable: PASS. Use the existing CPU notebook environment with no data or checkpoint download.
- Measure before making performance claims: PASS. Use paired starts, equal maximum transition budgets, and report success, steps, model calls, and runtime.
- Keep changes small and reviewable: PASS. Add one notebook and one report; do not add a planner framework or dependency.

## Design Decisions

- Use a point mass with velocity, bounded acceleration, and one rectangular obstacle. The goal lies on the other side and a valid trajectory must move around the obstacle.
- Use the same transition function and collision rule as the toy ground truth and planner model. State clearly that this removes learned-model error and does not create a latent representation.
- Implement simplified nearest-neighbor-guided RRT and projected-cell KPIECE selection. Both share one kinodynamic expansion procedure and one action set.
- Use categorical CEM over the same action set as a shooting baseline. Give every planner the same maximum number of one-step transition evaluations per decision; count and report actual calls because tree search can stop early.
- Replan from the current toy state after a fixed action prefix. Keep the tree local to each decision, matching the paper's stated re-grounding boundary without implementing its learned model or DINO-WM prior.
- Evaluate 20 shared starts in each of five declared evaluation-seed groups. Report variation across those groups; do not call it training-run variation.
- Report all planned episodes and missing/error counts. Keep raw per-episode records out of Git; the notebook regenerates them from fixed seeds.

## Project Structure

```text
specs/023-sampling-based-planning-tutorial/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/requirements.md
└── tasks.md
notebooks/21_sampling_based_planning.ipynb
docs/reproductions/sampling-based-planning.md
```

**Structure Decision**: Keep the lesson in one notebook and its summary in one report. Reuse the shared project evaluation protocol. Do not add source modules or a general-purpose planner library.
