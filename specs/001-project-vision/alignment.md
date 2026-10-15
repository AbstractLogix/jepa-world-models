# Alignment Review

**Reviewed**: 2026-10-15
**Method**: Manual check of the project spec, plan, tasks, README, curriculum guide, coverage map, evaluation protocol, and roadmap. Notebooks 01–08 and 10–17 have local CPU evidence; notebook 09's optional checkpoint path is recorded separately. Notebook 15's full 16-cell notebook run passed in a fresh CPU Jupyter kernel and is illustrative execution evidence. Notebooks 05 and 06 each completed twice with identical per-episode return lists. Notebook 13 completed ten local runs across five seeds for each method; the exact paper benchmark remains open. Notebook 14 records 60 action interventions and 20 paired policy starts in a local analogue. Notebook 16 records 60 held-out interventions and 20 paired starts for three methods in a rendered MountainCar analogue; its report applies the shared protocol and labels the result exploratory. Notebook 17 completes five local model runs and records calibration, selected-plan, and paired policy evidence; it is not a paper reproduction. The LeWAM host preflight is recorded in the [local report](../../docs/reproductions/lewam-pusht.md), but its asset transfer and evaluation remain pending. Feature 013 completed one full seed per model with the official K=0 probe and labels the result a partial reproduction attempt; the paper's three-seed result remains open.

| Requirement | Source of truth | Project evidence | Status |
| --- | --- | --- | --- |
| FR-001: State a clear path | `spec.md` | README and `CURRICULUM_GUIDE.md` now point to the same core sequence. | Aligned |
| FR-002: Map each current lesson | `spec.md` | `docs/CURRICULUM_COVERAGE.md` lists all nineteen notebooks and their limits. | Aligned |
| FR-003: Separate source claims and local results | Constitution and `spec.md` | Coverage and research documents label paper claims and local measurements separately; no local paper benchmark is claimed. | Aligned |
| FR-004/FR-005: Source and sequence future lessons | `spec.md` | `docs/ROADMAP.md` gives sources, outcomes, acceptance evidence, and priorities. Notebook 14 completes the decision-usefulness analogue; Notebook 15 teaches the LeHami structure with partial seed-0 evidence; Notebook 16 measures visual action-conditioned planning; Notebook 17 measures uncertainty calibration and selected-plan gaps across five local model runs. Notebook 19 teaches H-JEPA's top-down planning flow with a hand-written CPU analogy; its released-checkpoint Visual AntMaze evaluation remains blocked by renderer startup, including Mesa EGL and GLFW diagnostics. T041 scopes the PiJEPA reproduction. T046–T048 scope the Semigroup-JEPA candidate. T049–T052 scope the H-JEPA checkpoint evaluation and its access, license, and runtime gates. T053–T057 scope Qantara's three inference paths; Notebook 18 teaches the paths with a CPU analogy, and its optional evaluation remains blocked by unverified HDF5 lineage and renderer startup. A schema-compatible H-JEPA converter and a hash-matching staged DINO-WM archive are leads, not proof of Qantara's corpus identity. LeWAM, point-cloud, LeHami, Semigroup-JEPA, H-JEPA, and Qantara paper evaluations remain separately gated. | Aligned; current notebooks and research candidates are distinct |
| FR-006: Keep the core path independent of robot hardware | `spec.md` | Current control examples use Gymnasium CartPole and continuous MountainCar. No robot hardware is needed for the documented path. | Aligned; notebooks 10 through 12 and 14 runtime checked |
| FR-007: Track changes and gaps | Constitution and `spec.md` | Spec Kit 1.2.0 files and the feature task list are present. | Aligned |
| FR-008/FR-009: Link canonical docs and date research | `spec.md` | README links to coverage, evaluation protocol, and roadmap; research documents show a review date. | Aligned |
| FR-010 through FR-013: Scope paper-reproduction attempts | `spec.md` | `docs/REPRODUCTION_MATRIX.md` lists selected claims, official assets, access gates, and run status; `docs/EVALUATION_PROTOCOL.md` separates paper-faithful, scaled, and partial attempts and maps claims to the needed evidence. Notebook 13's Oxford-IIIT Pet study is labeled as a local transfer study, separate from the paper benchmark. Features 013 and 015 pin LeHami source/configs/data and report the seed-0 K=0 and separate K=1-fitted rollout measurements as partial evidence. | Aligned; exact paper benchmarks remain open |
| FR-014: Use a locked `uv` environment | `spec.md` | `uv.lock` resolves the PEP 621 project metadata and selects CPU PyTorch wheels; learner quickstarts document the optional CUDA 12.8 override. Constitution v2.0.0 records this boundary. T071 verifies the E:-hosted sync, imports, and RTX 5070 matrix check. | Aligned |

## Research roadmap refresh

The 2026-10-15 source review added LeWAM for RoboTwin 2.0 (arXiv:2609.27455), GA-WM (arXiv:2609.03565), and Hamiltonian JEPA (arXiv:2609.33497) to the future-work map. It names these separately from the existing Push-T LeWAM, diffusion-steering LeWAM, hierarchical H-JEPA, and LeHami entries. The released RoboTwin checkpoint still needs a fixed-case runtime check and a code-license review; the other two candidates still need first-party implementation and evaluation artifacts. No local results are claimed. See the [roadmap](../../docs/ROADMAP.md), [reproduction matrix](../../docs/REPRODUCTION_MATRIX.md), and [research review](research.md).

## Evaluation protocol audit

T062 checked the protocol fields and claim labels in the Notebook 16, Notebook 17, and LeHami reports. T064 extends the audit to Notebook 14: the saved outputs contain 60 intervention rows and 40 paired episode rows, and the report states its actual compute and evidence limits. The tracked Notebook 16 and 17 CSVs have the documented row counts and complete status values. The LeHami report records its source, data, checkpoint hashes, and pending seeds; its external raw files were not re-read. The audit confirmed report and record coverage, not model performance or a paper reproduction. See the [audit record](../../docs/evidence/evaluation-protocol-audit.md). T051's follow-up MuJoCo EGL, Mesa EGL, and GLFW diagnostics all stopped before a frame; the official checkpoint evaluation remains gated.

## Notebook 09 repeat verification

T063 reran the official V-JEPA 2.1 checkpoint path in a fresh CPU kernel. The notebook validated with no cell errors; its checkpoint hash and `[1, 8, 24, 24, 768]` feature shape matched the existing record. The synthetic patch check remains illustrative, not paper-benchmark evidence. See the [repeat-run record](../../docs/evidence/vjepa21-local-run.md).

## Latest roadmap update

JEPA-WAM is a pinned released-checkpoint evaluation candidate under feature 021. Its T066 smoke loaded the model on the RTX 5070, rendered one LIBERO spatial task, and ran ten no-op warm-up steps plus one policy action. The task returned 0/1; this is pipeline evidence, not a paper success rate. All required files and metadata are verified on E:. The full 2,402-task run remains pending after a naive first-task estimate of about 61.7 hours. The released training recipe differs from the paper, so any later result must be labeled as checkpoint re-evaluation.

## Requirement coverage

| Requirement | Task coverage |
| --- | --- |
| FR-001 | T006, T007 |
| FR-002 | T006, T028, T029 |
| FR-003 | T006, T008, T028, T029 |
| FR-004 | T008, T010, T011, T012, T013, T014, T015, T027, T028, T029, T031, T034, T035, T036, T037, T039, T041, T042, T043, T044, T046, T047, T049, T050, T053, T054, T055 |
| FR-005 | T008, T009, T027, T028, T029, T031, T034, T035, T036, T037, T038, T039, T042, T044, T046, T047, T049, T050, T053, T054, T055 |
| FR-006 | T006, T007, T014, T028, T043 |
| FR-007 | T001, T002, T003, T016, T042, T045, T054, T055 |
| FR-008 | T007 |
| FR-009 | T004, T006, T008, T044 |
| FR-010 | T017, T031, T032, T033, T051, T052, T053, T055, T057 |
| FR-011 | T017, T018, T031, T032, T033, T051, T052, T053, T055, T057 |
| FR-012/FR-013 | T017, T019, T020, T031, T032, T033, T035, T036, T049, T051, T052, T053, T055, T057 |
| FR-014 | T071 |
| SC-001 | T006, T007, T028, T029 |
| SC-002 | T007, T029 |
| SC-003 | T008, T031, T034, T035, T036, T043 |
| SC-004 | T006, T008, T028, T029, T031, T032, T033, T036, T043, T044 |
| SC-005 | T016, T035 |
| SC-006 | T009 through T015, T031 through T040, T042 through T045, T053 through T056 |
| SC-007 | T017 through T020, T022 through T026, T031, T032, T033, T036, T049 through T057 |
| SC-008 | T018 through T020, T031, T032, T033, T051 through T057 |
| SC-009 | T071 |
| US1 acceptance scenarios | T006, T007, T014 |
| US2 acceptance scenarios | T008 |
| US3 acceptance scenarios | T009 through T015, T037 through T041, T042 through T057, T063 |
| US4 acceptance scenarios | T001 through T005, T016 |
| US5 acceptance scenarios | T017 through T020, T022 through T029, T031 through T036, T049 through T057 |

The project vision now has 14 functional requirements and 9 success criteria. T053–T055 add a Qantara multi-paradigm tutorial and evaluation candidate. Notebook 18 is now a current CPU analogy, not a local paper reproduction. The released-checkpoint evaluation remains open. A hash-matching raw DINO-WM archive was staged before the Qantara task, and a pinned H-JEPA converter offers a compatible HDF5 schema; neither verifies Qantara's actual source episodes or row mapping. No Qantara-specific transfer or conversion occurred. Local evidence reports record CPU runs for notebooks 01–08 and 10–19; Notebook 09's optional checkpoint path was repeated in a fresh kernel under T063. Notebook 15's full notebook execution passed locally, while its structural example remains distinct from the partial seed-0 LeHami measurements. T036 adds one-seed K=1-fitted rollout measurements to the seed-0 K=0 results; the paper's three-seed result remains open. Notebook 13 completed its five-seed transfer study; the paired interval includes zero, and the paper-matched COCO result is still open. Notebook 14 completed the feature 012 analogue: 60 interventions and 20 paired starts; learned-model MPC succeeded on 0/20 starts and direct control on 20/20, with different actual compute. Notebook 16 completed 60 visual-feature interventions and 20 paired starts: neutral-start CEM succeeded on 0/20, while controller-prior CEM and direct control each succeeded on 20/20. Direct control reached the goal sooner on average and used no model calls. Notebook 17 completed five independent state-model runs, 60,000 calibration component rows, and 200 initial plan checks. Both learned planners reached 0/100 goals; direct momentum control reached 20/20. Its local state-model results do not reproduce a paper finding. Notebook results apply only to their stated MountainCar setups. Feature 013's seed-0 runs measured 0.456258 rad/s for LeHami and 2.504686 rad/s for LeWM using the official K=0 probe. The other four runs and the paper's three-seed result remain open. The LpWM PushT attempt was OOM-killed before training completed. LeWAM's preflight report records that Python 3.11 and MuJoCo/HDF5 support are absent from the current environment; transfer and evaluation remain pending. The initial point-cloud preflight failed CUDA and VRAM checks; the E:-hosted runtime now passes a basic CUDA check, but its 12 GB VRAM remains below the official 24 GB requirement. See the [gate report](../../docs/reproductions/point-cloud-pusht.md). Feature 017 records a scoped PiJEPA Table 2 comparison, but it is not run because CAST terms and trained checkpoints are unavailable and the current compute host is unqualified; see the [preflight report](../../docs/reproductions/pijepa-cast-planning.md). Semigroup-JEPA is a roadmap candidate for a gravity-conditioned square comparison; it is not implemented or reproduced, and its full comparison lacks several model artifacts and a qualified runtime. See the [preflight report](../../docs/reproductions/semigroup-jepa-preflight.md). V-JEPA Policy remains a research candidate because its trained checkpoints are not released. JEPA-WAM's one-task smoke is complete: one task rendered, ten no-op warm-up steps ran, one policy action ran, and the episode returned 0/1. The full 2,402-task evaluation remains pending, and no paper score is claimed locally.

No conflict was found between the current lessons and the remaining roadmap. Notebooks 04–08 and 11–19 are local analogues or transfer studies; none validates the cited paper results by itself. The LeWAM and point-cloud evaluations remain unrun. H-JEPA has a local CPU analogy, but its one-task evaluator attempt stopped at EGL initialization; OSMesa also fails before an environment starts. Exact paper reproductions and project-wide tutorial completion remain open in `tasks.md` and the reproduction matrix.
