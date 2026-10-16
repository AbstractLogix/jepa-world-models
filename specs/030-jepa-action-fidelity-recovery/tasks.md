# Tasks: JEPA Action Fidelity and Representation Recovery

**Input**: Design documents from specs/030-jepa-action-fidelity-recovery/  
**Prerequisites**: spec.md, plan.md, research.md, data-model.md, quickstart.md

## Phase 1: Audit and lock

- [x] T001 Recompute Notebook 23 v1 claims from docs/evidence/visual-jepa-decision-study-v1-*.csv
- [x] T002 [P] Record confirmed findings and hypotheses in docs/reproductions/visual-jepa-action-fidelity-v2-investigation.md
- [x] T003 [P] Freeze numeric gates, seeds, budgets, and stage rules in docs/reproductions/protocols/visual-jepa-action-fidelity-v2.md
- [x] T004 Create the Spec Kit feature specification, plan, research notes, data model, quickstart, and checklist in specs/030-jepa-action-fidelity-recovery/

## Phase 2: Diagnostic and positive-control implementation

- [ ] T005 Implement effective-rank, norm, gradient, action-sensitivity, state-clipping, and probe-conditioning diagnostics in src/jepa_tutorial/action_fidelity_v2.py
- [ ] T006 Implement validation-only objective/control training and fail-closed evidence writing in scripts/run_visual_jepa_action_fidelity_v2.py
- [ ] T007 Add exact, state-space, handcrafted visual, and supervised visual controls in scripts/run_visual_jepa_action_fidelity_v2.py
- [ ] T008 Add tests for finite/constant embeddings, action input sensitivity, probe conditioning, clipping, rollout validity, deterministic seeds, and split isolation in tests/test_visual_jepa_action_fidelity_v2.py
- [ ] T009 Implement v2 evidence completeness, integrity hashes, and gate validation in scripts/validate_visual_jepa_action_fidelity_v2.py
- [ ] T010 Run bounded CUDA validation only after the preregistration and implementation hashes are locked; save checkpoints only to E:

## Phase 3: Conditional held-out and planning evaluation

- [ ] T011 [US3] Implement independent action interventions and horizons 1, 5, 10, 20, and 60 in scripts/run_visual_jepa_action_fidelity_v2.py
- [ ] T012 [US3] Implement fail-closed validation and held-out gate transitions in scripts/validate_visual_jepa_action_fidelity_v2.py
- [ ] T013 [US3] Run held-out trajectories only if the supervised visual control and selected JEPA family pass validation
- [ ] T014 [US3] Run the locked CEM comparison only if held-out action and rollout gates pass; preserve all v1 starts and budgets

## Phase 4: Analysis, figures, and project alignment

- [ ] T015 [P] Generate separate deterministic figures A–F from retained records in scripts/analyze_visual_jepa_action_fidelity_v2.py
- [ ] T016 Analyze seed-level records and update docs/reproductions/visual-jepa-action-fidelity-v2-investigation.md
- [ ] T017 [P] Update docs/ROADMAP.md, docs/EVALUATION_PROTOCOL.md, and docs/REPRODUCTION_MATRIX.md
- [ ] T018 [P] Update docs/TECHNICAL_REPORT.md, docs/portfolio/CASE_STUDY.md, and docs/portfolio/project.json
- [ ] T019 Run unit tests, Ruff, package build, evidence validation, CPU smoke, figure regeneration, and git diff checks; record exact results
- [ ] T020 Open a draft PR based on PR #5 and attach it to this task; do not merge

## Dependencies

- Phase 2 depends on the frozen protocol and v1 audit in Phase 1.
- Phase 3 depends on complete validation records and passing gates. A failed gate closes later stages.
- Phase 4 can record a negative diagnostic report even when Phase 3 does not open.
- Publication depends on preserved original evidence and completed integrity checks.
