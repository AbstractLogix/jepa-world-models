---
description: "Tasks for the project vision, coverage, and research roadmap"
---

# Tasks: Project Vision, Curriculum Coverage, and Research Roadmap

**Input**: Design documents in `specs/001-project-vision/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Verification**: Notebooks 01–22 have local execution records or scoped reports. Notebook 22's protocol-matched VISReg-style CUDA pilot passed the fixed collapse screen. The full five-seed comparison then completed: 15/15 model fits, 960/960 intervention rows, and 440/440 episode rows. Uniform-start JEPA CEM and oracle-state CEM both failed on their fixed starts; JEPA CEM with the hand-written visual-momentum prior and direct visual-momentum control succeeded on all their starts. This is a local synthetic MountainCar comparison, not a paper reproduction or evidence that JEPA adds value beyond the prior. Notebook 09's official checkpoint path passed a fresh CPU-kernel rerun on 2026-10-15. Notebook 15's full 16-cell run passed in a fresh CPU kernel; Notebooks 16, 20, and 21 have paired evaluation evidence. The LpWM reproduction attempt stopped before epoch 1. Notebooks 05 and 06 have two matching seeded runs each. See `docs/evidence/`, `docs/reproductions/`, and `docs/CURRICULUM_COVERAGE.md` for scope and evidence limits.

**Organization**: Tasks follow the user stories in `spec.md`. Checked tasks are complete in this checkout. Open tasks remain part of the long-term project goal.

## Phase 1: Setup

**Purpose**: Add the shared specification workflow.

- [x] T001 Initialize Spec Kit 1.2.0 and Codex skills in `.specify/` and `.agents/skills/`
- [x] T002 Set project principles and governance in `.specify/memory/constitution.md`

## Phase 2: Foundational

**Purpose**: Record the project baseline and the evidence that guides future work.

- [x] T003 Create the feature specification, implementation plan, research notes, data model, and quickstart in `specs/001-project-vision/`
- [x] T004 Record alignment and remaining gaps in `specs/001-project-vision/alignment.md` and `specs/001-project-vision/remaining-work.md`
- [x] T005 Check the installed Codex integration with Spec Kit 1.2.0 in `.specify/integration.json`

**Checkpoint**: The project baseline is recorded. Existing notebook and source code remains intact.

## Phase 3: User Story 1 - Follow the current learning path (Priority: P1)

**Goal**: Learners can select current lessons and understand what each one demonstrates.

**Independent Test**: Check all current notebook entries and compare the path in the README and curriculum guide.

- [x] T006 Map all current notebooks, their topics, and their limits in `docs/CURRICULUM_COVERAGE.md`
- [x] T007 Align learning outcomes and prerequisites in `README.md` and `CURRICULUM_GUIDE.md`

## Phase 4: User Story 2 - Choose a future frontier lesson (Priority: P2)

**Goal**: Learners and instructors can choose future lessons from a dated, source-backed roadmap.

**Independent Test**: Check every roadmap item for a primary source, learner outcome, priority, and completion evidence.

- [x] T008 Review primary research sources and publish the sequenced roadmap in `docs/ROADMAP.md` and `specs/001-project-vision/research.md`

## Phase 5: User Story 3 - Study frontier topics with runnable lessons (Priority: P3)

**Goal**: Learners can study current video, language, and action-planning work through runnable examples.

**Independent Test**: Run one lesson with its stated data and model; record measured results, baseline, and run conditions.

- [x] T009 Define a shared evaluation protocol for seeds, task success, compute, variability, and data-source review in `docs/EVALUATION_PROTOCOL.md`
- [x] T010 Add a V-JEPA 2.1 dense-feature lesson using an official checkpoint; record source, license, data, hardware, and a local fallback in `notebooks/09_vjepa21_dense_features.ipynb`
- [x] T011 Add an action-conditioned planning lesson that compares open-loop plans, feedback MPC, and subgoals by goal distance and rollout horizon in `notebooks/10_planning_limits.ipynb`; see the [local evidence report](../../docs/reproductions/planning-limits.md). This is an illustrative analogue, not a paper reproduction.
- [x] T012 [P] Add real CIFAR-10 class-prompt embeddings, source and checkpoint details, and a linear-classifier comparison in `notebooks/11_vl_jepa.ipynb`; see the [local evidence report](../../docs/reproductions/vl-jepa.md). This is an analogue, not a paper reproduction.
- [x] T013 [P] Compare bounded continuous latent actions with discrete codes on generated clips; report same-view, new-shape, and new-camera results in `notebooks/12_video_latent_actions.ipynb`. See the [local run report](../../docs/reproductions/latent-actions.md). This is a synthetic analogue, not a paper reproduction.
- [x] T014 [P] Compare current teaching losses with paper-aligned LeJEPA and Rectified LpJEPA objectives across five seeds; record data, model scale, metrics, and limits in `notebooks/02_lejepa_from_scratch.ipynb`, `notebooks/03_rectified_lpjepa.ipynb`, and the [local run report](../../docs/reproductions/lejepa-rectified-lpjepa.md). This does not reproduce either paper's ImageNet benchmark.
- [X] T015 [P] Add an object-centric LeJEPA lesson that compares image-level and masked-object representations; state mask source and measure one downstream task in `notebooks/13_object_centric_lejepa.ipynb`. All ten Oxford-IIIT Pet runs and metrics are recorded in the [local report](../../docs/reproductions/object-centric-lejepa.md). The exact paper benchmark remains open.

## Phase 6: User Story 4 - Extend and check the project (Priority: P4)

**Goal**: Contributors can keep intended outcomes, implementation plans, and remaining gaps linked.

**Independent Test**: Check that this baseline's specification, plan, tasks, and remaining-work record describe the same change.

- [x] T016 Document the contributor workflow and its local setup in `README.md` and `specs/001-project-vision/quickstart.md`

## Phase 7: User Story 5 - Reproduce selected paper findings (Priority: P2)

**Goal**: Contributors can make bounded, evidence-backed attempts to repeat reported findings.

**Independent Test**: Each candidate has an exact claim, first-party sources, data and compute gates, and a not-run or local-evidence status. A completed report separates paper values from every local run.

- [x] T017 Define reproduction labels and source/access gates in `docs/EVALUATION_PROTOCOL.md` and `docs/REPRODUCTION_MATRIX.md`
- [x] T018 Specify a bounded PushT reproduction of one LpWM paper result after reviewing the official configs, data source, metric, and seeds in `specs/003-lpwm-pusht-reproduction/spec.md`
- [x] T019 Attempt the selected official LpWM configuration and record the incomplete local run, deviations, and compute in `docs/reproductions/lpwm-pusht.md`
- [x] T020 Update the paper-reproduction matrix with the attempt status and evidence in `docs/REPRODUCTION_MATRIX.md`

## Phase 8: Update the research and reproduction roadmap

**Goal**: Keep new papers, official artifacts, and resource limits visible in the future lesson plan.

**Independent Test**: Review new entries against their primary papers and official project repositories. Confirm that each status distinguishes paper claims, local evidence, and unavailable resources.

- [X] T021 Review October 2026 JEPA research and add distinct LeWAM, point-cloud JEPA, and JEPA-Anything candidates with source and artifact limits in `docs/ROADMAP.md` and `docs/REPRODUCTION_MATRIX.md`.
- [X] T022 Specify an exact released-checkpoint LeWAM Push-T evaluation with the paper's seeds and episode count in `specs/010-lewam-pusht-reproduction/`.
- [ ] T023 Run the official LeWAM Push-T evaluation for seeds 42, 0, and 1 and record all episodes, deviations, and local compute in `docs/reproductions/lewam-pusht.md`. Host preflight is recorded; Python 3.11 and MuJoCo/HDF5 support are absent, and archive transfer remains pending.
- [X] T024 Verify from the official evaluator that Point-cloud JEPA evaluation needs CUDA 12 and 24 GB GPU memory; create its bounded Push-T feature and pin the v1 data/checkpoint requirement in `specs/011-pointcloud-jepa-reproduction/` and `docs/REPRODUCTION_MATRIX.md`.

## Phase 9: Add decision-usefulness evidence to the roadmap

**Goal**: Ensure future lessons test whether world models support useful decisions, not only whether their features or predictions look accurate.

**Independent Test**: Check the added sources against their primary papers. Confirm the protocol names separate prediction, intervention, closed-loop, decision-benefit, and reliability evidence. Confirm the new paper-reproduction candidate preserves its seed and evaluation limits.

- [X] T025 Review recent decision-making evaluation research and add claim-to-evidence guidance to `docs/EVALUATION_PROTOCOL.md`, `docs/ROADMAP.md`, and `specs/001-project-vision/research.md`.
- [X] T026 Add the official JEPA-WM Push-T Table 2 comparison as a reproduction candidate; record its reported values, official artifacts, non-commercial license, and unresolved seed-artifact gate in `docs/REPRODUCTION_MATRIX.md`.

## Phase 10: Deliver the decision-usefulness lesson

**Goal**: Test whether a fitted world model supports counterfactual action predictions and useful closed-loop decisions against a direct control baseline.

**Independent Test**: The lesson reports held-out action interventions, 20 paired closed-loop starts, per-method compute, and explicit claim limits.

- [X] T027 Specify the decision-usefulness lesson, evidence measures, and resource limits in `specs/012-decision-usefulness/`.
- [X] T028 Implement and run Notebook 14; record the intervention and closed-loop results in `docs/reproductions/decision-usefulness.md`.
- [X] T029 Align the curriculum coverage, roadmap, project alignment, and remaining-work records with the completed lesson and evidence.
- [X] T030 Analyze and converge feature 012 after the implementation and local run.

## Phase 11: Reproduce action-conditioned symplectic latent dynamics

**Goal**: Compare the paper's LeHami and LeWM actuated-pendulum result with a bounded local run, then build a tutorial from the evidence.

- [x] T031 Specify the LeHami Table 1 K=0 comparison, pin official code/config/data revisions, and record the three-seed target in `specs/013-lehami-actuated-reproduction/`.
- [x] T032 Complete the selected full seed-0 training and K=0 25-NN evaluation for `a0q` and `b0s`; record per-run MAEs and compute in `docs/reproductions/lehami-actuated-velocity.md`.
- [ ] T033 Run seeds 1 and 2 for both models and report all six values plus the three-seed summary before describing the paper finding as reproduced.
- [x] T034 Add [Notebook 15](../../notebooks/15_lehami_symplectic_dynamics.ipynb), a mathematical analogue with partial paper-linked evidence, and record its limits and full local notebook execution in `docs/evidence/lehami-symplectic-tutorial.md`.
- [x] T035 Analyze and converge feature 013 after its selected run and alignment work; keep any remaining seeds as explicit open tasks.
- [x] T036 Evaluate the saved seed-0 checkpoints with the official K=1-fitted, frozen 25-NN rollout probe through K=76; report the held-out curve and limits while keeping the four additional training runs open in `specs/015-lehami-actuated-rollout/` and the local run report.

## Phase 12: Connect visual observations, action-conditioned prediction, and planning

**Goal**: Add a CPU-sized simulator analogue that makes the path from rendered frames to a predictive model and closed-loop planning measurable, while keeping exact paper claims open.

- [x] T037 Specify the Notebook 16 lesson, evidence, current research sources, and paper-reproduction boundaries in `specs/016-action-video-planning/`.
- [x] T038 Implement and execute Notebook 16; preserve the raw episode and intervention records and local results in `docs/evidence/` and `docs/reproductions/action-video-planning.md`.
- [x] T039 Align the README, curriculum guide, roadmap, coverage map, reproduction matrix, and project alignment records with the Notebook 16 evidence.
- [x] T040 Analyze and converge feature 016 after its implementation and documentation updates; keep paper-scale claims separate. No feature-scope gaps remained.
- [x] T041 Scope one exact policy-guided planning paper result for a future reproduction. Feature 017 pins the PiJEPA Table 2 target and records first-party code, data terms, checkpoints, evaluator, compute, and stop conditions in `specs/017-pijepa-cast-reproduction/` and `docs/reproductions/pijepa-cast-planning.md`. No data or model was downloaded and no run started.

## Phase 13: Measure uncertainty calibration and model-selected plan gaps

**Goal**: Add a bounded CPU lesson for uncertainty calibration and model-selected planning, then align local results with the research roadmap without claiming a JEPA or paper reproduction.

- [X] T042 Specify five independent state-model runs, disjoint simulator splits, calibration and planner comparisons, evidence records, and claim limits in `specs/018-model-uncertainty-planning/`.
- [X] T043 Implement and execute Notebook 17; preserve run, calibration, selected-plan, and episode records in `docs/evidence/` and report the local measurements in `docs/reproductions/model-uncertainty-planning.md`.
- [X] T044 Align the README, curriculum guide, evaluation protocol, roadmap, coverage map, reproduction matrix, project alignment, and remaining-work records with Notebook 17 evidence and its paper-reproduction boundary.
- [X] T045 Analyze and converge feature 018 after implementation and documentation updates; no feature-scope gaps remained.

## Phase 14: Research gravity-conditioned latent dynamics

- [x] T046 Review Semigroup-JEPA's paper claim, exact square evaluation protocol, first-party code and checkpoint artifacts, license notes, and local compute gate; record the findings in `docs/reproductions/semigroup-jepa-preflight.md`.
- [x] T047 Add Semigroup-JEPA to the research roadmap and reproduction matrix; align project research and remaining-work records while keeping it out of current notebook coverage.
- [ ] T048 Recheck asset terms, hashes, data split, and CUDA fit before the square comparison. Evaluate all four methods on the paper's 25 gravity values and 200 episodes per value at horizons 5, 16, 32, and 44. Find the missing SSM, LeWM, and DINO-WM artifacts or use an approved training path. Match the paper's model-seed count; if it is not stated, use five independent training seeds for locally trained methods or label a smaller checkpoint evaluation partial. Keep raw per-episode results outside Git and report episode variation separately from training-run variation.

## Phase 15: Research hierarchical visual planning

- [x] T049 Review H-JEPA's Visual AntMaze finding, official evaluator, checkpoint and task availability, licenses, and host requirements; record pins and limits in `docs/reproductions/h-jepa-antmaze-preflight.md`.
- [x] T050 Add H-JEPA to the future tutorial roadmap and reproduction matrix; align project research and remaining-work records while keeping the result separate from local evidence.
- [ ] T051 Review D4RL Ant-asset attribution and remaining dependency terms; verify selected checkpoint hashes; qualify isolated Python 3.11/CUDA 12.8 runtime with a one-task smoke for LeWM and H-JEPA seed 42. Record memory and wall time before any full evaluation. Current attempt: LeWM exits 139 during EGL initialization; OSMesa, Mesa EGL (surfaceless and WSLg Wayland/D3D12), and GLFW renderer-only checks also fail before a frame. H-JEPA smoke, memory, and runtime measurements remain open.
- [ ] T052 If T051 passes, evaluate both released models on all 50 official AntMaze tasks for seeds 42, 43, and 44. Preserve per-task results and report per-seed success and mean ± standard error. Label the result as a checkpoint evaluation, not an independent training reproduction.

## Phase 16: Research multi-paradigm JEPA inference

- [x] T053 Review Qantara paper v2, official code and checkpoint revisions, Push-T data sources, licenses, sizes, hashes, evaluator settings, and host renderer; record the findings and stop conditions in `docs/reproductions/qantara-preflight.md`.
- [x] T054 Specify the Qantara CPU tutorial and released-checkpoint Table 2 evaluation in `specs/019-qantara-multiparadigm/`. Keep the optional paper evaluation separate from the local teaching analogue.
- [x] T055 Align the README, roadmap, reproduction matrix, curriculum gap list, project research, alignment, remaining-work, and project tasks with the Qantara candidate; list Notebook 18 as a CPU tutorial, not as a paper reproduction or local paper result.
- [x] T056 Implement and execute the CPU Qantara inference-path tutorial in `notebooks/18_qantara_three_inference_modes.ipynb`; record its outputs and analogy limits in `docs/evidence/qantara-inference-tutorial.md`.
- [ ] T057 Resolve exact corpus identity and qualify the pinned runtime and renderer. If all gates pass, evaluate the three released checkpoint seeds with all three paths on five declared seeds and 50 episodes per seed. Preserve raw results outside Git and label changed sample identities.

## Phase 17: Teach hierarchical visual planning

- [x] T058 Specify the H-JEPA CPU tutorial in `specs/020-hjepa-hierarchical-planning/`; keep the learned-model evaluation separate.
- [x] T059 Implement and run `notebooks/19_hjepa_hierarchical_planning.ipynb` in a fresh CPU kernel; record its outputs and analogy limits in `docs/evidence/hjepa-hierarchical-planning.md`.
- [x] T060 Align the README, curriculum coverage, roadmap, reproduction matrix, project research, alignment, remaining work, and task ledger with the local lesson and gated Visual AntMaze evaluation.
- [x] T061 Analyze and converge the local tutorial feature after implementation and evidence review. Keep H-JEPA's released-checkpoint evaluation under T051–T052.

## Phase 18: Validate shared evaluation protocol

- [x] T062 Audit protocol field coverage and claim labels across the Notebook 16, Notebook 17, and LeHami reports; check tracked CSV row counts and status fields; record findings and clarify how shared run details may be reported in `docs/evidence/evaluation-protocol-audit.md`.

## Phase 19: Revalidate the official dense-feature path

- [x] T063 Repeat Notebook 09's opt-in V-JEPA 2.1 checkpoint path in a fresh CPU kernel; verify the checkpoint hash, feature shape, correspondence output, and error-free notebook; update the evidence and coverage records.

## Phase 20: Extend shared evaluation protocol coverage

- [x] T064 Audit Notebook 14's report fields, claim label, actual compute, and saved intervention and episode outputs against `docs/EVALUATION_PROTOCOL.md`; extend the audit and align the protocol and curriculum records.

## Phase 21: Scope the JEPA-WAM released-checkpoint evaluation

- [x] T065 Review the JEPA-WAM paper, pinned code and checkpoint revisions, LIBERO-Plus task count and protocol, third-party terms, and local host; create feature 021 and align the preflight, roadmap, and reproduction matrix.
- [x] T066 In an isolated Python 3.10 environment, verify the scoped inference artifacts and run the upstream one-task diagnostic with its ten no-op warm-up steps and one policy action. Record CUDA, memory, renderer, versions, elapsed time, and errors. Do not transfer training RLDS data. The one-task smoke completed with 0/1 task success; this is not a paper score. See the [evidence report](../../docs/reproductions/jepa-wam-libero-plus.md).
- [ ] T067 If T066 passes and the measured runtime is acceptable, evaluate all 2,402 spatial-suite variants across seven categories with one trial per task. Preserve each outcome and compute. Report this as a released-checkpoint re-evaluation, not a training reproduction.

## Phase 22: Refresh the public research map

- [x] T068 Review newly released JEPA world-model and world-action papers at arXiv:2609.27455, arXiv:2609.03565, and arXiv:2609.33497; verify first-party code, checkpoint, benchmark, and license status before adding candidates.
- [x] T069 Align the research roadmap, reproduction matrix, curriculum coverage, project research, alignment, and remaining-work records. Disambiguate the RoboTwin and Push-T LeWAM studies and hierarchical versus Hamiltonian H-JEPA. Keep all new paper claims separate from local evidence.
- [x] T070 Align the project plan and Spec Kit task records with the current count of nineteen notebooks.

## Phase 23: Adopt uv for the project environment

**Goal**: Use one fast, committed lockfile workflow for the tutorial environment and keep platform-specific PyTorch wheels documented separately.

**Independent Test**: `uv sync --locked` installs the project into the selected environment, the package imports, learner setup commands use `uv`, and no Poetry lockfile or CLI setup remains.

- [x] T071 Replace the Poetry lock and build backend with `uv.lock` and a standard build backend; align the constitution with CPU wheels in the lock and a separate CUDA override; update learner setup guides and external-drive paths; verify a clean E:-hosted environment.

## Phase 24: Deliver a policy-noise planning lesson

**Goal**: Add a source-backed, locally runnable lesson about planning in a generative policy's noise space while keeping the paper reproduction separate.

- [x] T072 Deliver feature 022's CPU-only Notebook 20 and local report; align the README, curriculum coverage, roadmap, reproduction matrix, project research, and remaining-work record.
- [x] T073 Validate the shared evaluation protocol against Notebook 20's fresh-kernel run and update `docs/EVALUATION_PROTOCOL.md`, `docs/evidence/evaluation-protocol-audit.md`, and the project alignment record.

## Phase 25: Deliver a sampling-based planning lesson

**Goal**: Teach tree search and shooting optimization with a reproducible toy task while keeping the paper evaluation separate.

- [x] T074 Deliver feature 023's CPU-only Notebook 21 and local report; align the README, curriculum, roadmap, reproduction matrix, project research, and remaining-work record.
- [x] T075 Apply the shared evaluation protocol to Notebook 21, extend the protocol audit, and analyze and converge feature 023. Keep the paper reproduction not run until author-linked artifacts are available.

## Phase 26: Align the shared protocol with Notebook 09 evidence

**Goal**: Record the fresh V-JEPA 2.1 checkpoint inspection as a bounded illustrative measurement in the shared protocol and its audit.

**Independent Test**: The protocol, audit, curriculum coverage, roadmap, and project alignment all describe the same Notebook 09 evidence and state that it does not establish general feature quality or reproduce a paper benchmark.

- [x] T076 Add Notebook 09 to the protocol audit and align its evidence boundary across the protocol, curriculum, roadmap, and project records.

## Phase 27: Align retained episode evidence

**Goal**: Keep the shared protocol, research roadmap, and project records aligned with the saved Notebook 20 and 21 episode rows.

**Independent Test**: The canonical protocol and roadmap link both CSV files. The project alignment and remaining-work records state the row coverage and preserve the illustrative claim boundary.

- [x] T077 Update the protocol and roadmap with the Notebook 20 and 21 episode CSV links; update the project alignment and remaining-work records without presenting toy outcomes as paper evidence.

## Phase 28: Validate retained episode evidence

**Goal**: Make the integrity checks for the saved Notebook 20 and 21 episode records repeatable.

**Independent Test**: The documented command passes both current CSVs, reports their 300 rows and complete paired coverage, and rejects temporary copies with missing or malformed evidence. The protocol audit states that this validates record integrity only.

- [x] T078 Add and run the episode evidence validator, document its command and limits, and align the evaluation protocol audit and project status records. See [feature 026](../026-episode-evidence-validation/spec.md).

## Phase 29: Preflight LeWAM RoboTwin 2.0

**Goal**: Establish source, terms, and host readiness before any LeWAM RoboTwin evaluation assets are transferred.

**Independent Test**: The preflight records the reviewed paper and code revision, reports no local run, identifies the WSL/runtime and license gates, and aligns the roadmap, matrix, and project records. It remains separate from LeWAM Push-T.

- [x] T079 Record the source-pinned LeWAM RoboTwin preflight and align the roadmap, reproduction matrix, project alignment, and remaining-work record. See [feature 027](../027-lewam-robotwin-preflight/spec.md) and the [preflight report](../../docs/reproductions/lewam-robotwin-preflight.md).

## Phase 30: Review calibrated JEPA risk screening

**Goal**: Add a recent, decision-focused safety paper to the research map without presenting author-reported results as local evidence or treating the learned model as a safety guarantee.

**Independent Test**: The research review, tutorial roadmap, reproduction matrix, curriculum coverage, project alignment, and remaining-work inventory agree on the paper's scope, simulation-only evidence, unresolved reranking comparison, source status, and no-run status.

- [x] T080 Review arXiv:2608.17496 and align it as a gated P3 watch item. Record the reported LIBERO-Long comparisons, the nonsignificant reranking-only margin, compute and execution gaps, and that the reviewed sources do not expose a first-party location for the release materials. No assets were transferred and no evaluation was run.

## Phase 31: Learned visual JEPA decision usefulness

**Goal**: Measure when an action-conditioned learned visual representation supports action selection, while preserving model failures and compute conditions.

**Independent Test**: A non-collapsed visual JEPA passes the fixed validation screen before the predeclared held-out interventions and paired episode comparison run. All five training seeds, raw evaluation rows, model access differences, and compute limits are recorded.

- [x] T081 Specify and implement the local visual-JEPA lesson, model primitives, optional CUDA selection, CPU fallback, fixed protocol, validation-only pilot mode, and hash-matched fail-closed gate in feature 028. The lesson is not a paper reproduction.
- [x] T082 Run and record validation-only CUDA pilots on the RTX 5070, including the full 16-episode training and 4-episode validation split. The default SIGReg model and separate VICReg-style objective probe both collapsed across all 32 dimensions; no intervention or evaluation starts were used. See [feature 028](../028-visual-jepa-decisions/spec.md) and the [partial report](../../docs/reproductions/visual-jepa-decision-usefulness.md).
- [x] T083 Continue the collapse diagnosis. The GPU replay found a zero SIGReg input gradient for exact constant zero embeddings but no confirmed defect or validated fix. Review the VISReg paper equations and current first-party source/license availability; no implementation or license was available, so no code or weights are reused. Update feature 028's spec, research, and plan before any new objective pilot. Do not inspect intervention or evaluation outcomes.
- [x] T084 Implement the independent VISReg-style temporal-prediction loss, CPU checks, objective selector, evidence schema, and matching pilot gate. Preserve SIGReg and historical records.
- [x] T085 Run one fixed-split, 35-epoch validation-only CUDA pilot on seeds 1000/2021. Record settings, components, device, time, memory, and collapse status. Do not collect held-out outcomes. The pilot passed the fixed spread screen.
- [x] T086 After the matching pilot passed, train the replacement final pairs 7221/5100, 7227/5101, 7239/5102, 7253/5103, and 7263/5104 for each learned baseline. Record prediction, spread, device, time, and memory. Exclude pre-gate pairs 2121/1100, 2127/1101, 2139/1102, 2153/1103, and 2163/1104.
- [x] T087 After training, run the fixed held-out action interventions and paired closed-loop conditions. Preserve all 960 intervention rows and 440 episode rows. Label privileged state access, native scoring spaces, and compute; all rows completed without errors.
- [x] T088 Complete the feature report, align the README, curriculum, roadmap, reproduction matrix, and project records with the measured results, then run focused checks and Spec Kit analysis/convergence. The report records 24 passing unit tests, Ruff, package build, notebook and evidence checks, and a clean Spec Kit review; the full CPU comparison and paper-scale benchmarks remain unrun.

## Phase 32: Complete the preregistered matched visual JEPA study

**Goal**: Finish Notebook 23's locked local comparison, publish its complete evidence with a bounded claim, and preserve the remaining learned-policy research gap.

**Independent Test**: The evidence validator passes for the complete episode matrix and retained source hashes. The report, portfolio manifest, curriculum, protocol, reproduction matrix, roadmap, project alignment, and task ledger agree on the counts, low-spread warning, model/data seed pairing, missing prior-only planner, handcrafted direct controller, and paper-reproduction boundary.

- [x] T089 Complete the locked CUDA episode matrix, validate all v1 evidence, and retain the report and corrected paired analysis. Preserve the low-spread status as partial even if every episode record completes.
- [x] T090 Add reproducible accessible study figures and update the case study, project manifest, README, curriculum, protocol, reproduction matrix, roadmap, and remaining-work record using only validated results.
- [ ] T091 Run focused tests, CPU smoke, Ruff, package build, notebook/evidence/figure/link checks, and Spec Kit analysis/convergence; review the diff and publish a draft PR without merging.

## Dependencies and Execution Order

- Setup and foundational tasks are complete.
- User Stories 1 and 2 are complete for the documentation baseline.
- T009 through T015 and T017 are complete. T018 through T020 record the incomplete LpWM reproduction attempt; its training and planner results remain open. T021, T022, and T024 add current candidates and bounded LeWAM and point-cloud reproduction features; T023 remains open pending a supported environment, data transfer, and evaluation. T025 and T026 add current decision-usefulness research and an official JEPA-WM Push-T candidate. T027 through T030 deliver and converge the local decision-usefulness lesson and alignment. The point-cloud run is open because this host fails its 24 GB CUDA gate. Exact benchmark reproductions for LeJEPA, Rectified LpJEPA, and object-centric LeJEPA remain open in `docs/REPRODUCTION_MATRIX.md`.
- T018 through T020 form the first paper-reproduction attempt. T018 must be complete before final evaluation runs; T019 must precede the evidence status update in T020.
- T031 specifies the LeHami actuated-pendulum comparison. T032 completes the user-selected seed-0 pair; T034 delivers Notebook 15 and its partial evidence record. T036 adds a separate seed-0 K=1-fitted rollout curve. T033 remains required for the paper's three-seed estimate; T010 in feature 013 records the four pending runs. T035 completed Spec Kit analysis and convergence for feature 013. Do not call the paper finding reproduced until T033 is complete.
- T037 through T040 specify, implement, measure, align, and converge the Notebook 16 action-conditioned video planning analogue. Its fixed color-moment model ranked 60/60 non-tied action pairs correctly, but neutral-start CEM succeeded on 0/20 starts; controller-prior CEM and direct control each succeeded on 20/20. Direct control reached the goal sooner on average and used no model calls. T042 through T044 specify, implement, execute, and align the Notebook 17 uncertainty-calibration and selected-plan-gap analogue; T045 runs its final Spec Kit analysis and convergence. Notebook 17's results apply only to its stated MountainCar setup. T041 is complete as a scoped preflight: feature 017 targets PiJEPA Table 2 and documents missing CAST terms and trained checkpoints, an unknown paper case list, and the host's compute mismatch. The paper reproduction remains blocked and unrun.
- T046 and T047 complete a source-backed Semigroup-JEPA preflight and roadmap alignment. T048 remains gated by missing comparison checkpoints or an approved training path, terms, data, and CUDA runtime. The candidate is not a current notebook or a local reproduction.
- T049 and T050 complete the H-JEPA preflight and roadmap alignment. T051 remains open: seed-42 checkpoint/task hashes and the Python 3.11/CUDA 12.8 environment are verified, but the official one-task LeWM smoke exits 139 during EGL initialization and OSMesa fails before an environment starts; the H-JEPA smoke has not run. T052 remains open and no AntMaze task outcome exists. The candidate is a released-checkpoint evaluation, not independent training evidence.
- T053 through T056 complete the Qantara research preflight, feature specification, CPU tutorial, and roadmap alignment. T057 remains gated: Qantara's actual HDF5-to-paper-corpus mapping is unverified, the paper's five evaluation-seed identities are unpublished, and the current host has no working renderer. A hash-matching raw DINO-WM archive was already staged outside this task; no Qantara-specific asset was transferred or converted, and no paper-evaluation result exists.
- T058 through T061 complete the H-JEPA CPU teaching analogy and align it with the future evaluation roadmap. Notebook 19 does not train or evaluate H-JEPA. T051 and T052 remain open for the separate Visual AntMaze checkpoint evaluation; no H-JEPA task result is available.
- T062 audited the protocol against three local reports and their tracked records. The audit found no missing or failed CSV rows and confirmed that the exploratory, simulator-analogue, and partial-reproduction labels match their evidence. It checks records only; it does not rerun results or verify external LeHami outputs.
- T064 extends the protocol audit to Notebook 14. Its saved notebook outputs contain 60 intervention rows and 40 paired episode rows; the report labels the work illustrative and states the difference between a shared time cap and actual compute. The paired intervals do not measure training-run variation.
- T063 repeated Notebook 09's official checkpoint path in a fresh CPU kernel. The checkpoint hash and feature shape matched the record, and the notebook completed without cell errors. This confirms only the local synthetic-clip path; it is not a V-JEPA 2.1 benchmark reproduction.
- User Story 4 has the Spec Kit workflow in place. Future feature work must use it and keep the roadmap aligned.

T065 pins the JEPA-WAM checkpoint evaluation. T066 must pass before T067 starts; the full task count and checkpoint-versus-paper training mismatch stay visible in any report.
T068 and T069 refresh the dated research map and public documentation. They add future candidates only; no model, dataset, or simulator assets were transferred and no new paper result was measured.
T070 corrects stale notebook counts in the project plan and alignment record; it does not change lesson scope.
T071 migrates the project Python environment to uv and amends constitution version 2.0.0 to record the CPU-lock/CUDA-override boundary. The E:-hosted Python 3.13 environment installs from the lock and imports the project, PyTorch, TorchVision, TorchCodec, and timm. The CUDA 12.8 override passes a matrix operation on the RTX 5070. The pre-existing repository `.venv` remains intact.
T072 adds one local analogy and its evidence; it does not reproduce the LeWAM diffusion-steering paper. T073 applies the existing protocol to its paired starts, compute record, and claim boundary. The five seed groups are evaluation variation, not model-training runs.
T074 adds a second planning analogy: 100 shared starts and five evaluation groups for RRT, KPIECE, and CEM. Its success, transition-call, runtime, and failure counts apply only to the known toy point-mass task. T075 applies the protocol and convergence workflow; the paper evaluation remains not run.
T076 adds Notebook 09's fresh checkpoint-path evidence to the protocol inventory. The comparison uses one synthetic clip and has no pass threshold; it remains illustrative and does not establish general feature quality or a paper benchmark.
T079 records a source-pinned LeWAM RoboTwin 2.0 preflight only. WSL GPU simulation/rendering is unsupported by RoboTwin, the CUDA toolkit and Vulkan tools are absent, and the code terms are unresolved. No assets or code were transferred; the fixed-case evaluation remains not run.
T080 adds a research watch item only. It does not implement a safety shield, evaluate a JEPA, or reproduce the cited simulation results.
T081 and T082 specify Notebook 22, implement the protocol-hash-matched pilot gate, and record the earlier SIGReg and VICReg-style collapse findings. A gradient replay found exact-collapse zero gradient but no confirmed fix. Five earlier model/data pairs remain excluded pre-gate diagnostics. T083 updates the Spec Kit design from VISReg equations; no paper code or weights are reused. T084–T085 add and run one validation-only VISReg-style temporal-prediction pilot, which passes the fixed spread screen. T086–T087 complete the replacement five-seed model fits, held-out interventions, and paired planner evaluation. The local results remain exploratory and do not reproduce any paper. T088 aligns the report, project docs, evidence, and Spec Kit analysis/convergence.

### Parallel opportunities

T015 has a defined mask source, one downstream task, and a complete local five-seed execution. Exact benchmark reproductions remain separate entries in `docs/REPRODUCTION_MATRIX.md`. T010 and T011 share the V-JEPA planning path and should follow the stated dependency order.

The local action-conditioned lesson does not reproduce a JEPA method. Do not begin a PiJEPA-scale run until the open access and compute gates in feature 017 pass. T041 scopes and records those gates; it does not authorize a transfer or count as a paper reproduction.

## Implementation Strategy

The documentation baseline and evaluation protocol are in place. The long-term goal remains open until future lessons have local execution evidence and selected paper findings have bounded reproduction attempts. Do not treat paper-reported results as local results.

