# Implementation Plan: Preflight LeWAM RoboTwin 2.0

**Branch**: `027-lewam-robotwin-preflight` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

## Summary

Record a source-pinned readiness review for the released LeWAM checkpoint on RoboTwin 2.0. The current WSL2 host is not suitable for RoboTwin GPU simulation or rendering, CUDA toolkit and Vulkan tools are absent, and the LeWAM code repository has no declared project license. No evaluation assets or code are transferred. Keep the candidate in the future roadmap and separate from the pending Push-T evaluation.

## Technical Context

- **Language/Version**: Documentation only; no runtime code changes
- **Primary Dependencies**: None
- **Storage**: No new data or model assets; record current free space on E: and D:
- **Testing**: Check local Markdown links, source revision references, task alignment, and whitespace
- **Target Platform**: WSL2 host with an RTX 5070; the RoboTwin guide does not support WSL GPU simulation or rendering
- **Project Type**: Research preflight documentation
- **Performance Goals**: No evaluation run in this feature
- **Constraints**: Do not clone or run the research code, install its environment, or transfer model or simulator assets while source terms and runtime gates remain unresolved
- **Scope**: Read-only source and host review; no paper result is measured

## Constitution Check

- **Teach from evidence**: Pass. Cite the paper and first-party implementation, model, and simulator sources.
- **Label teaching examples honestly**: Pass. The report labels 92.28% as a paper value and records no local result.
- **Make the learning path usable**: Pass. The future restart path uses one fixed case list before any broad evaluation.
- **Keep environments and assets controlled**: Pass. The report records the E:/D: capacity and confirms no evaluation assets were transferred.
- **Avoid unsupported conclusions**: Pass. A one-task fixed-case check is not a reproduction of the 50-task paper average.

## Design

The work changes documentation only. `research.md` stores source facts and decisions. `data-model.md` defines the small evidence records used in the report. `quickstart.md` describes how to review the preflight and its restart gates. The report is the canonical local status. The project roadmap, matrix, alignment, remaining-work, and task list link to it.

## Project Structure

```text
specs/027-lewam-robotwin-preflight/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md

docs/reproductions/lewam-robotwin-preflight.md
```

## Risks and Open Questions

- The official code and predictor terms are not declared in the reviewed repository metadata. Resolve those terms before depending on or redistributing that code.
- RoboTwin's official guide lists WSL as unsupported for GPU simulation and rendering. A future evaluation needs a supported native Linux environment or another explicitly supported host.
- RoboTwin asset size and license must be reviewed before transfer.

## Constitution Check After Design

Pass. The plan makes no runtime change, keeps source facts separate from local evidence, and sets explicit stop gates.
