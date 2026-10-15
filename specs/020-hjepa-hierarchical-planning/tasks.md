# Tasks: H-JEPA Hierarchical Planning Tutorial

**Input**: Design documents in this feature directory.

## Phase 1: Research and lesson design

- [x] T001 Review the H-JEPA v1 paper, pinned official code, current preflight, and project constitution; record the claim boundary in `research.md`.
- [x] T002 Define the synthetic teaching entities and local CPU scope in `data-model.md`, `quickstart.md`, and `spec.md`.

## Phase 2: User Story 1 - Understand top-down hierarchical planning (Priority: P1)

**Independent Test**: Run every notebook code cell on CPU. Confirm that the example produces a top-down subgoal chain and labels all toy mappings as hand-authored.

- [x] T003 Create `notebooks/19_hjepa_hierarchical_planning.ipynb` with the paper-backed method summary, level-specific toy summaries, top-down subgoal chain, and explicit limits.
- [x] T004 Execute every notebook cell in a fresh CPU kernel and record versions, outputs, and claim limits in `docs/evidence/hjepa-hierarchical-planning.md`.

## Phase 3: Alignment and review

- [x] T005 Align the README, curriculum coverage, roadmap, project research, alignment, remaining work, and project task ledger with the local lesson and gated evaluation status.
- [x] T006 Analyze the feature artifacts against the implementation and confirm that no additional tutorial work is missing. Keep the independent Visual AntMaze evaluation tasks open.

## Dependencies

- T001 and T002 define the evidence and teaching boundary before notebook implementation.
- T003 precedes T004.
- T005 follows local execution evidence.
- T006 follows implementation, execution, and alignment.
