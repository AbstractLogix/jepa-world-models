---
description: "Tasks for a version-pinned Push-T point-cloud JEPA evaluation"
---

# Tasks: Point-Cloud JEPA Push-T Reproduction

**Input**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md` in this directory.

**Verification**: On a supported CUDA host, run both released models for the ten paper seeds and 50 episodes per seed. Verify asset hashes, paired windows, and all raw results. Record evidence in `docs/reproductions/point-cloud-pusht.md`.

## Phase 1: Preflight and pinned assets

### User Story 1 - Re-run the matched Push-T comparison (Priority: P1)

- [X] T001 [US1] Check Linux, CUDA 12, EGL, and 24 GB GPU memory before data transfer; record the host gate and result in `docs/reproductions/point-cloud-pusht.md` (FR-010). Host preflight failed the CUDA runtime and GPU memory gates; later tasks remain stopped before transfer.
- [ ] T002 [US1] Prepare the official `fafraob/point-lewm` checkout at commit `1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424` in a temporary external work directory (FR-001, FR-009).
- [ ] T003 [US1] Install the locked upstream environment and build its point-cloud kernels in the temporary checkout; record software versions and build status in `docs/reproductions/point-cloud-pusht.md` (FR-008).
- [ ] T004 [US1] Download the Push-T v1 archive and two model folders at the revisions in `specs/011-pointcloud-jepa-reproduction/research.md`; verify all three SHA-256 values before evaluation (FR-002, FR-003, FR-009, SC-005).

## Phase 2: Matched checkpoint evaluation

### User Story 1 - Re-run the matched Push-T comparison (Priority: P1)

- [ ] T005 [US1] Run Point-LeWM with the official `lidar_pusht` config for all ten seeds and 50 episodes per seed; save raw results outside Git and record them in `docs/reproductions/point-cloud-pusht.md` (FR-004, FR-005, FR-006, SC-001, SC-002, SC-003).
- [ ] T006 [US1] Run Point-Delta-JEPA with the same config, seed list, and 50 episodes per seed; save raw results outside Git and record them in `docs/reproductions/point-cloud-pusht.md` (FR-004, FR-005, FR-006, SC-001, SC-002, SC-003).
- [ ] T007 [US1] Verify both methods have 500 episode outcomes, ten rates, and the same 50 start-goal windows for each seed in `docs/reproductions/point-cloud-pusht.md` (FR-005, FR-006, SC-001, SC-002).

## Phase 3: Evidence and curriculum alignment

### User Story 2 - Keep the reproduction auditable (Priority: P1)

- [ ] T008 [US2] Calculate per-method mean and standard deviation across the ten seed rates; compare the means with the paper's one-standard-deviation ranges; record the descriptive check and point-resampling variation in `docs/reproductions/point-cloud-pusht.md` (FR-007, FR-008, SC-003, SC-004).
- [ ] T009 [US2] Align `docs/CURRICULUM_COVERAGE.md`, `docs/REPRODUCTION_MATRIX.md`, `docs/ROADMAP.md`, `specs/001-project-vision/alignment.md`, and `specs/001-project-vision/remaining-work.md` with the report, resource limits, and claim boundaries (FR-009, SC-005).
- [ ] T010 [US2] Run Spec Kit analysis and convergence for `specs/011-pointcloud-jepa-reproduction/spec.md`, `plan.md`, and `tasks.md` after the evaluation tasks are complete.

## Dependencies and Execution Order

- T001 is a gate. If the host fails, stop before T002–T006 and record the blocker in `docs/reproductions/point-cloud-pusht.md`.
- Complete T002 and T003 before T004.
- Complete T004 before T005 and T006. Run T005 and T006 sequentially on the same host.
- Complete T005 and T006 before T007 and T008. Complete T007 and T008 before T009 and T010.

## Implementation Strategy

1. Check the hardware gate before downloading the multi-gigabyte dataset.
2. Pin and verify the upstream code, v1 dataset, and released model files.
3. Run both methods on paired paper seeds and preserve every episode outcome.
4. Record measured values and deviations. Keep the reproduction status open if a full ten-seed run cannot finish.
