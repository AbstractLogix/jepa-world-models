# Tasks: Qantara Multi-Paradigm Inference Tutorial and Evaluation

**Input**: Design documents in `specs/019-qantara-multiparadigm/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, and quickstart.md

**Execution gate**: The CPU tutorial is independent. Do not transfer Push-T assets or start official evaluation until data identity, serialization, storage, and renderer gates pass.

## Phase 1: Research and preflight

**Purpose**: Pin the paper target and record access and runtime limits.

- [x] T001 Review current Qantara paper v2, code revision, three Push-T checkpoint records, official configs, metric, and evaluation counts in `research.md` and `docs/reproductions/qantara-preflight.md`.
- [x] T002 Review checkpoint/data license declarations, Hub sizes and hashes, Push-T dataset-source mismatch, and current host renderer result; do not transfer artifacts.
- [ ] T003 Resolve the lineage of the actual Qantara HDF5 corpus against the paper's DINO-WM Push-T corpus. Compare source episode IDs, counts, sequence lengths, transformations, row order, and hashes; record the five local evaluation seeds and a verified conversion before any evaluation.
- [ ] T004 Review the pinned checkpoint loader and serialization behavior; record a controlled loading procedure and confirm the selected artifact and expanded data sizes fit available storage. Qualify the pinned runtime and headless renderer with a one-case smoke; record software versions, hardware, time, peak memory, and output before any full transfer.

**Checkpoint**: T003 and T004 are hard gates for evaluation. The current renderer gate fails; no full transfer or evaluation has started.

## Phase 2: User Story 1 - Explain three inference paths (Priority: P1)

**Goal**: Teach the same-checkpoint planning, BC, and video–inverse paths without external assets.

**Independent Test**: Run all notebook cells on CPU and check that path inputs, goal access, and analogy limits are explicit.

- [x] T005 Create `notebooks/18_qantara_three_inference_modes.ipynb` with the bridge-flow query map, three inference paths, a small CPU teaching analogue, source links, and paper/local evidence separation.
- [x] T006 Run the notebook in a fresh CPU kernel; record versions, settings, output, and scope limits in `docs/evidence/qantara-inference-tutorial.md`.

## Phase 3: User Story 2 - Evaluate the released checkpoints (Priority: P2)

**Goal**: Check Table 2's three Push-T inference paths on the same Qantara checkpoints and cases.

**Independent Test**: A reviewer can inspect all 2,250 episode rows across the three modes and three training seeds, with no missing case hidden from the summary.

- [ ] T007 After T003 and T004 pass, transfer only the verified Push-T corpus and Qantara checkpoints; verify local hashes and keep all binaries outside Git.
- [ ] T008 Run CEM K=4, BC K=2, and video–inverse K=2 for 50 episodes on each of five declared evaluation seeds for each Qantara checkpoint seed 11, 22, and 33, using the official 50-step budget and shared case sets.
- [ ] T009 Preserve all raw episode outcomes, case identities, checkpoint and evaluation seeds, method, software and hardware, timing, peak memory, failures, and deviations outside Git. Summarize paper values separately from per-training-seed local success and population standard deviation across the three seed means in `docs/reproductions/qantara-pusht.md`.
- [ ] T010 Label any changed evaluation seed, failed episode, missing checkpoint, or data deviation as partial; do not call an unmatched sample-level result exact.

## Phase 4: Evidence and alignment

- [x] T011 Align README, curriculum coverage, roadmap, reproduction matrix, project research, alignment, remaining work, and project tasks with the actual tutorial and run status.
- [ ] T012 Run Spec Kit analysis after tasks and tutorial implementation; append convergence tasks only after implementation and evidence review.

## Dependencies and execution order

- T001 and T002 are complete research work. A pinned H-JEPA converter is a schema-compatible candidate for DINO-WM data, but no first-party record ties it to Qantara's actual HDF5 file. A hash-matching raw DINO-WM archive was already staged outside this task; it was not transferred, unpacked, or converted here. T003 and T004 remain runtime/data gates.
- T005 and T006 can proceed without the paper evaluation and do not depend on T003/T004.
- T007 through T010 require both gates and run sequentially from asset verification through reporting.
- T011 follows the notebook evidence and the actual reproduction status.
- T012 follows implementation and alignment; do not converge an unimplemented feature.
