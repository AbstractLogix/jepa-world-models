# Implementation Plan: Point-Cloud JEPA Push-T Reproduction

**Branch**: `011-pointcloud-jepa-reproduction` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification in `specs/011-pointcloud-jepa-reproduction/spec.md`

## Summary

Evaluate the official Point-LeWM and Point-Delta-JEPA Push-T checkpoints with the paper's v1 data and ten-seed protocol. Use the upstream evaluator in an isolated checkout. Keep all downloaded assets and run outputs outside this project. Record every episode and seed result in one report.

## Technical Context

**Language/Version**: Python 3.10 (pinned upstream environment)  
**Primary Dependencies**: The official `pixi.lock` environment; PyTorch 2.7 with CUDA 12.6; no new project dependency  
**Storage**: Upstream checkout and generated outputs in a temporary or external work directory; Push-T v1 archive is 4,616,741,240 bytes; project retains only a Markdown report  
**Testing**: Verify source and asset revisions and hashes; run the official evaluator; verify 50 outcomes for every method and seed; recompute per-seed rates and ten-seed summaries  
**Target Platform**: Linux x86-64, NVIDIA GPU with CUDA 12, EGL, and at least 24 GB GPU memory  
**Project Type**: Research evaluation and tutorial evidence report  
**Performance Goals**: Complete 1,000 episodes across two methods and ten paired seeds; record wall time per seed and total  
**Constraints**: Do not train models; use paper-matched v1 data; stop before data download if the host fails preflight; do not commit data, checkpoints, upstream code, or generated media  
**Scale/Scope**: One Push-T result from Table 2; two point-cloud methods; ten seeds; 50 episodes per seed

## Constitution Check

- **Teach from evidence**: Pass. The source paper and official code, dataset, and model pages are listed in `research.md`.
- **Label teaching examples honestly**: Pass. The report is an exact released-checkpoint evaluation attempt, not a training reproduction or physical-robot result.
- **Make the learning path usable**: Pass with a clear resource gate and a stop-before-download rule.
- **Measure before making performance claims**: Pass. Both methods use paired seeds and report raw outcomes and seed-level spread.
- **Keep changes small and reviewable**: Pass. No model code, project dependency, or shared runtime is added.

## Design

1. Run preflight on the intended Linux host. Require CUDA 12, EGL, and 24 GB GPU memory before downloading data.
2. Use the official code at `1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424`. This commit is dated 2026-10-12. Its changes affect image data/evaluation paths; the point-cloud evaluator and Push-T LiDAR config remain the selected path.
3. Fetch the v1 Push-T archive at `d3391d60371b6e9cbcd1c5737bdfc87b4af309ed` and the two checkpoint folders from model revision `664278a4c0b4c5f607d0993744986b2d10b4168e`. Verify each selected file's SHA-256 before evaluation.
4. Use the official `lidar_pusht` evaluation: 300 CEM samples, 30 iterations, horizon 5, frame skip 5, goal offset 25, 50 environment steps, and 50 episodes per seed. Disable videos to reduce storage.
5. Run both methods for seeds `1, 25, 35, 36, 40, 42, 65, 80, 83, 99`. Preserve one raw record per episode and one success rate per seed.
6. Report the ten-seed mean and standard deviation for each model. Compare the means to the paper values and their one-standard-deviation ranges. State that this check is descriptive, not a statistical test.
7. Write `docs/reproductions/point-cloud-pusht.md`; update the coverage map, reproduction matrix, and project alignment status. Keep source assets and outputs external to the repository.

## Project Structure

### Documentation (this feature)

```text
specs/011-pointcloud-jepa-reproduction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

### Evaluation artifacts

```text
<external-work-dir>/point-lewm/                 # pinned upstream checkout
<external-work-dir>/point-lewm/data/             # paper-matched Push-T v1 data
<external-work-dir>/point-lewm/checkpoints/pusht/ # selected released checkpoints
<external-work-dir>/point-lewm/eval_results/      # upstream logs, no videos
docs/reproductions/point-cloud-pusht.md          # compact retained evidence
```

**Structure Decision**: Keep the source checkout and large assets outside this project. Retain only the report and status links in this repository.

## Constitution Check After Design

Pass. The design keeps paper claims separate from local measurements, uses the exact checkpoint-matched dataset revision, records resource limits, and does not add or vendor external code.

## Complexity Tracking

No constitution violations.
