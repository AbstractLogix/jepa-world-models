# Implementation Plan: Qantara Multi-Paradigm Inference Tutorial and Evaluation

**Branch**: `019-qantara-multiparadigm` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/019-qantara-multiparadigm/spec.md`

## Summary

Add a local CPU tutorial that explains Qantara's three inference paths, plus a separate official checkpoint evaluation of the paper's Push-T Table 2 result. The tutorial uses no external model or data. The checkpoint evaluation remains gated because the Push-T corpus mapping is not confirmed, the paper's five evaluation-seed identities are unpublished, and the current host's MuJoCo EGL and OSMesa paths fail.

## Technical Context

**Language/Version**: Project Python 3.13 for the local notebook; the official Qantara project supports Python 3.10 through 3.13.

**Primary Dependencies**: Existing notebook dependencies for the local tutorial. The optional evaluation uses the pinned Qantara repository, PyTorch 2.7.x, stable-worldmodel, MuJoCo 3.2.0, the official checkpoints, and the verified Push-T corpus.

**Storage**: Keep checkpoints, data, and raw episode outputs outside Git. The DINO-WM Push-T archive is 2.79 GB compressed; the directly compatible LeWM HDF5 archive is 13.14 GB compressed. Confirm the selected format, expansion size, and available space before transfer.

**Testing**: Run every local notebook cell in a fresh CPU kernel. If all evaluation gates pass, run the official evaluator and preserve raw per-episode output. No separate test suite is requested.

**Target Platform**: CPU for the tutorial. Optional evaluation needs Linux, a supported CUDA GPU, and a working headless MuJoCo renderer.

**Project Type**: Tutorial notebook and research evidence documentation; no runtime library changes.

**Performance Goals**: No performance claim for the local analogy. The optional run reports local latency and memory separately from the paper's H100 timings.

**Constraints**: Do not transfer Push-T data or checkpoints or start evaluation until source data identity, checkpoint provenance, serialization handling, storage, CUDA, and rendering are verified. Do not call a run exact if it uses different evaluation-seed identities.

**Scale/Scope**: One environment, three Qantara inference paths, three released training checkpoints, five declared evaluation seeds, and 50 episodes per evaluation seed. Do not add training or other environments.

## Constitution Check

- Teach from evidence: PASS. The paper and official repository are pinned and linked; no local Qantara result is claimed before measurement.
- Label teaching examples honestly: PASS. The CPU lesson is an analogy, separate from the checkpoint evaluation.
- Make the learning path usable: PASS. The notebook does not require a GPU, robot, data, or model weights.
- Measure before making performance claims: PASS. The optional evaluation names the task, modes, settings, run count, and required raw outputs.
- Keep changes small and reviewable: PASS. One notebook, one report, and alignment edits; no new project dependency or library abstraction.

**Gate result**: The local tutorial can proceed. The official evaluation is blocked until the data mapping, exact source artifacts, and headless rendering gates pass.

## Project Structure

```text
specs/019-qantara-multiparadigm/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── tasks.md
└── checklists/requirements.md
notebooks/18_qantara_three_inference_modes.ipynb
docs/reproductions/qantara-preflight.md
docs/evidence/qantara-inference-tutorial.md
docs/reproductions/qantara-pusht.md
```

**Structure Decision**: Keep the CPU teaching path in one notebook and keep its execution record separate from the paper-linked evaluation report. Do not add a code package, dependency, or asset to the core path.

## Complexity Tracking

No constitution violations or new runtime complexity.
