# Implementation Plan: Released LeWAM Push-T Checkpoint Reproduction

**Branch**: `010-lewam-pusht-reproduction` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

## Summary

Re-evaluate the official LeWAM Push-T checkpoint with the authors' evaluation code, dataset, settings, and three seeds. Run the reactive policy, best-of-32 planner, and gradient planner for 50 episodes per seed. Compare local rates with the paper's reported means. This work does not train LeWAM.

## Technical Context

**Language/Version**: Python 3.11 as requested by the official installer  
**Primary Dependencies**: The pinned LeWAM repository environment, PyTorch, MuJoCo, Gymnasium, HDF5 support, and the official stable-worldmodel environment  
**Storage**: Keep the code checkout, 13.8 GB compressed dataset, expanded 43.1 GiB HDF5 file, 64 MB checkpoint, logs, videos, and temporary results outside the project repository  
**Testing**: Preflight environment and artifact hashes; execute the full official evaluator; check that all 450 episode results are present and summarize by seed  
**Target Platform**: Linux x86-64; official evaluator code has a CPU fallback, but the runtime and throughput must be checked before the full run  
**Project Type**: Isolated scientific reproduction using official research code  
**Performance Goals**: Complete three evaluation modes × three seeds × 50 episodes, or record the exact blocking condition and keep the attempt partial  
**Constraints**: Do not train the model; do not add dependencies to project environment files; do not commit external artifacts; disclose that episodes are drawn from training data  
**Scope**: Official LeWAM Push-T checkpoint and reported goal-reaching protocol only

## Constitution Check

- **Teach from evidence**: Pass. Link the paper, official project page, code, model card, and dataset card. Separate reported and local values.
- **Label teaching examples honestly**: Pass. This is a released-checkpoint evaluation, not a training reproduction or held-out test.
- **Make the learning path usable**: Pass. Keep the research environment isolated and state its 13.8 GB / 43.1 GiB data needs before download.
- **Measure before making performance claims**: Pass. Use the official three seeds and 50 episodes per mode and seed. Preserve raw episode results.
- **Keep changes small and reviewable**: Pass. Only add a feature specification, compact report, and matrix/coverage links. Keep downloaded artifacts outside the repository.

## Research Decisions

1. Pin the official code at `60732e25ba172b117d0d0a476d3a9dcd5d30c03b`.
2. Pin `LeWAM/lewam-pusht` at revision `d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2`. Verify `lewam_best.pt` SHA-256 `e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb`.
3. Pin dataset `LeWAM/lewam-pusht` at revision `63346c94f4dbe62b4744261c6c99d63cbff3f6e6`. Verify `pusht.h5.zst` SHA-256 `a075dcd49120844a1720cd3f650519f934a632b5c3e9a5a4e413ba8b020aaa43`.
4. Set `STABLEWM_HOME` to a temporary or external path. Keep the official clone and generated logs out of the tutorial repository.
5. Run the exact three evaluation modes from the official reproduction guide for seeds 42, 0, and 1. Use the official Push-T goal offset of 25 steps and budget of 50 actions.
6. Use the HDF5 data path or the official default only after checking its memory and disk needs. Do not create the 352 GB image-decompression cache for evaluation unless the official config requires it.
7. Save the report at `docs/reproductions/lewam-pusht.md` and update the reproduction matrix only after the actual run or a specific failure is recorded.
8. Compare each local mode mean with the paper's reported mean ± one reported standard deviation. Label the result within or outside that range; do not treat this threshold as a statistical test.

## Project Structure

### Project documentation

```text
specs/010-lewam-pusht-reproduction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/requirements.md
└── tasks.md

docs/reproductions/lewam-pusht.md
```

### External run files

```text
/tmp/jepa-lewam-pusht/
├── code/        # detached official code checkout
├── data/        # official dataset and decompressed HDF5 file
├── checkpoints/ # released checkpoint and config
└── results/     # raw evaluator outputs and videos
```

**Structure Decision**: Use the official repository as the evaluator. This project records the protocol and measured evidence; it does not copy or vendor the research implementation.

## Constitution Check After Design

Pass. The plan uses official artifacts, keeps the run isolated, records raw outcomes, and limits conclusions to the released checkpoint and its sampled training episodes.

## Complexity Tracking

No constitution violations. The large public dataset and separate runtime are the main costs; both are visible before execution.
