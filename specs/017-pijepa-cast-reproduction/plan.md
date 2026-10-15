# Implementation Plan: PiJEPA CAST Table 2 Reproduction

**Branch**: `017-pijepa-cast-reproduction` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/017-pijepa-cast-reproduction/spec.md`

## Summary

Scope one exact PiJEPA Table 2 result: V-JEPA-2 ViT-L, with PiJEPA, Octo-WM, Octo, and default MPPI compared on CAST validation episodes. The current work is a source and access preflight. The run is blocked because CAST terms are not stated, the reviewed PiJEPA source does not provide the required trained checkpoints, the paper does not publish its exact validation-case list, and the available GPU does not match the paper's four-H200 setup. The official wrapper also defaults to one Octo sample although the paper uses four, and its evaluator does not clip the initial policy-prior standard deviation as the paper specifies. Do not transfer the dataset or start training or evaluation until these gates pass or each deviation is explicitly resolved and recorded.

## Technical Context

**Language/Version**: Python 3.10 per the official PiJEPA README; local run environment is not qualified for this workload.
**Primary Dependencies**: Pinned PiJEPA repository, its official evaluation wrapper, V-JEPA-2 ViT-L, CAST data, and the authors' policy and world-model checkpoints. Exact install lock and compatible versions must be confirmed from the pinned source before a run.
**Storage**: Keep data and checkpoints outside Git. CAST archives total 80.8 GB before extraction; plan for expanded data, checkpoints, outputs, and temporary files before transfer. Store raw run records outside Git in a durable location that reviewers can access, and record its path and hashes in the report.
**Testing**: Manual source and artifact preflight; then the pinned official evaluator and raw per-case records after all gates pass. No local test suite or training run is added by this planning feature.
**Target Platform**: Linux with the paper-compatible four-H200 setup for a paper-aligned run. Current host has one RTX 5070 with 12 GB VRAM and is not qualified.
**Project Type**: Research and tutorial documentation; later evaluation uses the authors' Python repository.
**Performance Goals**: Match the paper's Table 2 target and evaluator settings. No local performance claim is made at preflight.
**Constraints**: No dataset or model transfer until written data terms, checkpoint provenance, evaluation cases, storage, and compute are verified. Use all four modes: default_mppi (Default MPPI), octo_mppi (PiJEPA), octo_wm (Octo-WM), and octo_mean (Octo). The paper states four Octo samples for the PiJEPA policy prior, but the wrapper applies one N_OCTO_SAMPLES value across modes and octo_mean averages all generated samples. Confirm the standalone Octo sample count before choosing per-method invocations. The current official evaluator passes the raw policy-sample standard deviation to the first MPPI iteration without clipping it to the paper's [0.01, 0.05] bounds; resolve this difference or report it as a deviation before a paper-aligned claim. Use at least five independent training seeds for stochastic training when the paper gives no run count; a smaller run stays partial or exploratory.
**Scale/Scope**: One encoder variant and one Table 2 row across four methods. Do not expand to other encoders, ablations, or tables in this feature.

## Constitution Check

- Evidence: PASS. The paper result and local status are in separate fields and cite primary sources.
- Honest labels: PASS. This feature is a blocked preflight, not a reproduction.
- Usable learning path: PASS. The quickstart states prerequisites and stop conditions; no hardware or private-data requirement is implied for the core tutorial path.
- Measured claims: PASS. The spec names methods, metric, split, seeds, and raw records; none is reported as measured locally.
- Small changes: PASS. This feature adds documentation and one bounded future run plan. No dependencies, notebook code, or downloaded files are added.

**Gate result**: Planning may proceed. Training and evaluation remain blocked by unresolved data terms, missing paper checkpoints, and an unqualified compute host.

## Project Structure

```text
specs/017-pijepa-cast-reproduction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
docs/reproductions/pijepa-cast-planning.md
docs/ROADMAP.md
docs/REPRODUCTION_MATRIX.md
specs/001-project-vision/{alignment.md,remaining-work.md,tasks.md}
```

**Structure Decision**: This is a documentation-only preflight. It adds no runtime code or external interface, so no contract directory is needed. If a later implementation requires code, create a new bounded plan after the access and compute gates pass.

## Complexity Tracking

No constitution violations or new runtime complexity.
