# Implementation Plan: H-JEPA Hierarchical Planning Tutorial

**Branch**: `020-hjepa-hierarchical-planning` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a CPU-only lesson that teaches H-JEPA's level-specific latent predictions and top-down subgoal flow with a small hand-authored example. Keep it separate from the released-checkpoint evaluation in the project task ledger.

## Technical Context

**Language/Version**: Python 3.10–3.13 in the existing notebook environment.
**Primary Dependencies**: Python standard library only.
**Storage**: No external assets. Save the notebook and its execution record in the repository.
**Testing**: Execute every notebook code cell in a fresh CPU kernel and inspect the claim labels and local links.
**Target Platform**: Jupyter on CPU.
**Project Type**: Tutorial notebook and evidence documentation.
**Constraints**: No H-JEPA checkpoint, dataset, GPU, or renderer use. Do not report the paper result as local evidence. Keep notebook order and current project dependencies unchanged.

## Constitution Check

- Teach from evidence: PASS. The paper result and method claims link to the v1 paper and pinned official code.
- Label teaching examples honestly: PASS. The toy mapping is explicitly hand-authored.
- Make the learning path usable: PASS. The lesson requires no external assets or GPU.
- Measure before performance claims: PASS. The toy reports no benchmark or performance comparison.
- Keep changes small and reviewable: PASS. One notebook, one evidence report, and index updates.

## Project Structure

```text
specs/020-hjepa-hierarchical-planning/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── tasks.md
└── checklists/requirements.md
notebooks/19_hjepa_hierarchical_planning.ipynb
docs/evidence/hjepa-hierarchical-planning.md
```

**Structure Decision**: Keep the educational example in one notebook. Do not add a model package, new dependency, or paper-evaluation code.
