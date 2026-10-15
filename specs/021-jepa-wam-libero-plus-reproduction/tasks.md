# Tasks: JEPA-WAM LIBERO-Plus Checkpoint Re-evaluation

**Input**: Design documents in this feature directory.

## Phase 1: Research and scope

- [x] T001 Review the paper, pinned author source, checkpoint card, LIBERO-Plus protocol, third-party terms, and local host; record the release mismatch and bounded target in research.md and docs/reproductions/jepa-wam-libero-plus.md.

## Phase 2: Qualify the released evaluation path

- [x] T002 In a disposable checkout and environment, verify only the required inference artifact hashes and run the upstream one-task diagnostic with its ten no-op warm-up steps and one policy action. Record dependencies, CUDA, GPU memory, renderer status, elapsed time, and errors. Do not download RLDS training data. The diagnostic rendered one task and executed one policy action; 0/1 task success is not a benchmark result. See the [evidence report](../../docs/reproductions/jepa-wam-libero-plus.md).

## Phase 3: Re-evaluate the published result

- [ ] T003 If T002 passes and the measured runtime is acceptable, evaluate the complete 2,402-task LIBERO-Plus spatial suite, all seven categories, one trial per task. Preserve per-task outcomes and compute in docs/reproductions/jepa-wam-libero-plus.md or a linked evidence file.
- [ ] T004 Compare the per-category and overall results with the paper and author README; state the 40,000-step checkpoint versus 60,000-step paper-training difference and update the roadmap, matrix, curriculum coverage, project alignment, and remaining work.

## Dependencies

- T001 is complete and defines the artifact and claim boundary.
- T002 must pass before T003 starts.
- T003 must finish before T004 can report a local score.
