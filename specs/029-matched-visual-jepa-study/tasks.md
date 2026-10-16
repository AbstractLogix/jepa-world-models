# Tasks: Matched Visual JEPA Decision Study

**Input**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`, and the preregistration.

## Phase 1: Specification and preregistration

**Purpose**: Lock the methodology and preserve the Notebook 22 boundary before any tuning.

- [x] T001 Audit Notebook 22, model/loss code, records, evaluation protocol, and reproduction matrix. Record observed settings, score-space mismatch, missing planner tuning, seed separation, and claim limits in `research.md`.
- [x] T002 Write the v1 preregistration with the planner search grid, disjoint seeds, compute limits, gate rules, primary outcome, and analysis before running a planner search.
- [x] T003 Run Spec Kit cross-artifact analysis. Resolve any critical requirement or task coverage gaps before code changes.

## Phase 2: Deterministic primitives and evidence contracts

**Purpose**: Make planner behavior, state splits, and gates testable on CPU.

- [x] T004 Add tests for MountainCar transition parity, CEM bounds, deterministic seeds, call counts, terminal behavior, and invalid planner settings.
- [x] T005 Implement pure transition, policy, CEM, task-score, seed-split, and budget helpers in `src/jepa_tutorial/decision_study.py`.
- [x] T006 Add tests and validators for lock/gate JSON, per-method episode contracts, model-seed separation, paired interventions, finite values, and failure rows.
- [x] T007 Implement fail-closed protocol-lock and oracle-gate checks. Smoke mode must not read or write final case files.

## Phase 3: Oracle planner development gate

**Purpose**: Tune a bounded planner only on development starts and lock it before final cases.

- [x] T008 Add the `dev-search` CLI mode with the preregistered two-stage grid, exact dynamics, deterministic CEM seeds, full setting records, and the 500M call ceiling.
- [x] T009 Add the tuning tie-break rules, disjoint confirmation set, 18/20 overall and 9/10 confirmation thresholds, code hash, and immutable protocol-lock output.
- [x] T010 Run the development search with the prepared CUDA/GPU environment where useful and the exact dynamics. Do not access final evaluation starts.
- [x] T011 Validate and retain every attempted setting. If no setting passes, record a failed development gate and skip final evaluation.

## Phase 4: Held-out exact-dynamics gate

**Purpose**: Confirm that the locked planner works on the held-out starts before evaluating learned models.

- [x] T012 Add `oracle-gate` mode. It must verify the preregistration and planner-lock hashes and write exact-dynamics rows only.
- [x] T013 Run the locked exact planner on the 20 final starts. Continue only if it succeeds on at least 18 starts; otherwise preserve the oracle rows and stop the learned-model stage.

## Phase 5: Matched learned-model study

**Purpose**: Measure model and planner-initialization value with common outcomes.

- [x] T014 Add training and data contracts for five shared random-trajectory groups with separate train, validation, and representation-test episodes.
- [x] T015 Train the state, visual reconstruction, and action-conditioned JEPA models for all five declared seeds. Reuse the matched data per seed and record device, software, objective, failures, time, and memory.
- [x] T016 Fit frozen visual position/velocity probes from training data, select regularization on validation data, and report only on test trajectories.
- [x] T017 Add held-out action interventions with the common continuation-return ranking and separate position/velocity prediction fidelity.
- [x] T018 Evaluate exact, state, reconstruction, and JEPA models under uniform and momentum-prior CEM on shared starts with matched budgets. Record direct visual control once per start.
- [x] T019 Preserve every episode, failed model, timeout, missing case, call count, elapsed time, planning time, access label, and seed.

## Phase 6: Report and figures

**Purpose**: Make evidence understandable and regenerate figures without training.

- [x] T020 Add a bounded tutorial notebook `notebooks/23_visual_jepa_decision_study.ipynb` that explains protocol gates and runs the versioned CLI.
- [x] T021 Add one deterministic A–D evidence dashboard and one E architecture diagram. Keep historical Notebook 22 outcomes separate, label the prior-only CEM arm not run, and fail closed on incomplete v1 evidence.
- [x] T022 Write the versioned report with results or gate status, per-seed records, intervals, compute, limits, negative findings, and primary-source links.
- [x] T023 Validate evidence schemas, row counts, pairing, hashes, missing runs, and deterministic figure inputs. Add a corrected hierarchical bootstrap summary that resamples shared evaluation starts once per replicate.

## Phase 7: Project alignment and delivery

**Purpose**: Align public docs and prepare a reviewable draft PR.

- [x] T024 Update README, curriculum guide/coverage, evaluation protocol, reproduction matrix, roadmap, portfolio case study/manifest, and the project-vision alignment, remaining-work, and task ledger only with verified study status and findings.
- [x] T025 Run focused unit and CPU smoke checks, Ruff, package build, notebook, evidence, figure, Markdown-link, and diff checks. Record each pass, failure, and not-run check in the report.
- [x] T026 Run Spec Kit convergence against the completed implementation and tasks. Append any remaining implementation work, complete it, and repeat until converged.
- [ ] T027 Review the final diff for historical evidence preservation and claim scope, push the focused branch, and open a draft PR. Do not merge.

## Dependencies and execution order

- T001–T003 precede implementation.
- T004–T007 establish deterministic contracts before the dev-search CLI.
- T008–T011 complete before any protocol lock or final start access.
- T012–T013 complete before learned-model training or held-out evaluation.
- T014–T019 run only after a passing oracle gate.
- T020–T025 use the actual evidence status and outputs.
- T026 runs only after implementation. Complete any convergence tasks before T027.

## Stop rules

- A failed development gate stops the study before final starts.
- A failed final oracle gate stops the study before learned-model evaluation.
- A compute-ceiling or model-collapse stop preserves partial rows and does not lower the preregistered seed or episode threshold.
