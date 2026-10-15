# Implementation Plan: Bounded LpWM PushT Reproduction

**Branch**: `003-lpwm-pusht-reproduction` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Reproduce one author-configured D=768 PushT comparison from the LpWM paper.

## Summary

Use the pinned upstream LpWM repository to train the sparse LpWM and dense LeWM MLP∘LTV models at D=768 on the same public PushT archive. Evaluate each checkpoint with the upstream closed-loop CEM-MPC planner for 50 episodes at seeds 99, 100, and 101. Keep source code, data, checkpoints, and raw output under `/tmp`. Add one evidence report and update the reproduction matrix. Label the attempt partial because it covers one latent dimension and one training seed.

## Technical Context

**Language/Version**: Python 3.9.25 for upstream; tutorial project remains Python 3.10–3.13.

**Primary Dependencies**: Upstream LpWM at revision `bdd812d9432cccda8c350086006401b436f91982`; PyTorch 2.3.0; torchvision 0.18.0; Accelerate and dependencies from the upstream environment file. Install CPU PyTorch wheels in an isolated `/tmp/lpwm-venv`.

**Storage**: OSF PushT archive and extracted files in `/tmp/lpwm-data`; checkpoints and logs in `/tmp/lpwm-runs`. No experiment artifacts enter version control.

**Testing**: Verify archive byte count and SHA-256; dry-run the filtered author grid; run a CPU import/model preflight; inspect training and planner outputs; validate all reported episode counts and arithmetic from raw run records. No new automated test suite is needed for this documentation-only change.

**Target Platform**: Local Linux CPU host; no CUDA device.

**Project Type**: Research reproduction documentation and evidence capture. No tutorial runtime code changes.

**Performance Goals**: Complete two training jobs and six planning runs if the CPU preflight is viable. Capture elapsed time and peak memory when available.

**Constraints**: Upstream code is designed for GPU use and its environment file includes CUDA packages. CPU wheels are a local deviation. OSF exposes the dataset but its metadata has no license field; do not redistribute the archive. The local CPU run was OOM-killed while loading the full training set with 20 data-loader workers. A full attempt needs a safe memory plan or a GPU-capable host; a CPU smoke check alone does not establish feasibility.

**Scale/Scope**: One sparse/dense pair, one training seed per arm, three planner seeds per arm, 50 trajectories per planner run. This is not a full grid or a multi-training-seed comparison.

## Constitution Check

- Teach from evidence: paper, source, and data links are recorded; paper values and local values are separate. **Pass.**
- Label the result honestly: one D=768 setting and one training seed are labeled a partial reproduction attempt. **Pass.**
- Make the learning path usable: this feature adds a report and matrix row; it does not add a learner-facing notebook. **Pass.**
- Measure before making claims: baseline, task, metric, configuration, seeds, and a directional decision rule are set before evaluation. **Pass.**
- Keep changes small: do not modify upstream source, core package dependencies, or tutorial code. **Pass.**
- Project evaluation protocol: the paper specifies one training seed, so the attempt follows that count. Three planner seeds do not count as independent training runs. The result cannot support a general performance claim. **Pass with explicit scope limit.**

## Workflow

1. Keep the author checkout at the reviewed commit and verify the filtered grid prints exactly the two intended cells.
2. Download the named OSF archive to `/tmp`; verify the listed size and SHA-256 before extraction. Preserve the unknown-terms status in the report.
3. Build the isolated Python 3.9 CPU environment. Confirm framework versions, CPU device, imports, dataset paths, and the selected model path before training.
4. Run the two-cell official script with `RUN=1`, `ARM_LIST="sparse dense"`, `PRED_LIST=ltv`, `PD_LIST=768`, `EPOCHS=2`, `NEVALS=50`, `MAXITER=10`, and temporary checkpoint storage. Save command output and logs under `/tmp`.
5. For each successful checkpoint, run the same planner configuration for seeds 100 and 101. The official filtered script supplies seed 99. Do not change planner settings between model arms.
6. Compute per-seed success rates, means, standard deviations, and sparse-minus-dense differences from the run outputs. Verify the decision rule from the pre-run research note.
7. Write `docs/reproductions/lpwm-pusht.md` with source, archive verification, conditions, raw per-run outcomes, compute, failures, deviations, and the evidence label. Update `docs/REPRODUCTION_MATRIX.md` to the observed status.

If the CPU preflight or either training run fails, keep the exact failure evidence and mark subsequent dependent work incomplete. Do not substitute published checkpoints or silently change the model, dataset, or budget.

## Project Structure

### Documentation (this feature)

```text
specs/003-lpwm-pusht-reproduction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
docs/
├── REPRODUCTION_MATRIX.md
└── reproductions/
    └── lpwm-pusht.md
```

**Structure Decision**: This is an evidence-only feature. The upstream experiment stays in a temporary checkout and environment. The repository stores only the reviewable method and results.

## Complexity Tracking

No constitution violations. No project dependencies or application code are added.
