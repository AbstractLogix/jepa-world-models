# Feature Specification: Project Vision, Curriculum Coverage, and Research Roadmap

**Feature Branch**: `001-project-vision` (spec directory only; existing `main` branch is preserved)
**Created**: 2026-10-14
**Status**: Baseline documentation complete; tutorial implementation remains open
**Input**: User request to add Spec Kit, align and converge the project, set a goal to fully implement its vision, research current work for future tutorials, and try to reproduce findings from the cited papers.

## User Scenarios & Testing

### User Story 1 - Follow the current learning path (Priority: P1)

A learner wants to study JEPA world models in a clear order. They need to know what each notebook teaches and what it does not show.

**Why this priority**: The existing notebooks are the main product. Learners need accurate outcomes and limits before using them.

**Independent Test**: A learner can choose a starting notebook and explain whether its results are a concept demo, a small experiment, or a paper result.

**Acceptance Scenarios**:

1. **Given** the README and curriculum guide, **When** a learner chooses a core lesson, **Then** both documents give the same path and learning outcome.
2. **Given** a lesson inspired by a paper, **When** a learner reads its coverage entry, **Then** the entry states its data, environment, and reproduction boundary.

### User Story 2 - Choose a future frontier lesson (Priority: P2)

A learner or instructor wants to study newer JEPA work in video, language, action learning, or planning. They need current sources and a clear reason for each proposed lesson.

**Why this priority**: The research is moving quickly. A dated, source-backed roadmap helps the course grow without overstating results.

**Independent Test**: A reader can select a roadmap item and find its primary source, learning outcome, dependency, and evidence needed to call the lesson complete.

**Acceptance Scenarios**:

1. **Given** the roadmap, **When** a contributor proposes a new lesson, **Then** the source and proposed local evidence are clear.
2. **Given** a paper result, **When** it appears in project documentation, **Then** it is labeled as a reported result until this project measures it.

### User Story 3 - Study frontier topics with runnable lessons (Priority: P3)

A learner wants to inspect current video and object features, vision-language prediction, or action-conditioned planning with runnable examples and honest limits.

**Why this priority**: These topics complete the project's stated path from representation learning to predictive control, but they need more data, model, and evaluation work than the current core lessons.

**Independent Test**: A learner can run one frontier lesson, identify the checkpoint or data it used, and report a measured outcome with its task and conditions.

**Acceptance Scenarios**:

1. **Given** a lesson that uses a released model, **When** a learner follows its local path, **Then** the model source, checkpoint, data, and hardware needs are stated before loading it.
2. **Given** a planning lesson, **When** a learner changes the rollout horizon or goal distance, **Then** the lesson records success against a named baseline and explains the feedback setting.

### User Story 4 - Extend and check the project (Priority: P4)

A contributor wants to add a tutorial without losing the link between the project goal, the requirements, and the current code.

**Why this priority**: A shared workflow helps contributors keep lessons, evidence, and future plans aligned.

**Independent Test**: A contributor can review this baseline and confirm that its specification, plan, tasks, and remaining-work record describe the same bounded change.

**Acceptance Scenarios**:

1. **Given** a proposed tutorial change, **When** the contributor records its learner outcome and acceptance criteria, **Then** the task list can be checked against those criteria.
2. **Given** a change that is not fully implemented, **When** the contributor reviews its status, **Then** missing work remains visible as open tasks.

### User Story 5 - Reproduce selected paper findings (Priority: P2)

A contributor wants to check whether a published finding appears under a clearly scoped local run. They need an exact claim, official sources, original setup, baseline, metric, and a record of deviations before they run it.

**Why this priority**: Reproduction attempts can test the project's implementations and make the lessons more useful. They also need tighter evidence rules than illustrative tutorials.

**Independent Test**: Every reproduction candidate has a cited claim, source code and data status, a bounded run plan, and a local-evidence status. A completed attempt reports every run and deviation. No paper result is called reproduced from a toy or scaled example.

**Acceptance Scenarios**:

1. **Given** a paper claim selected for study, **When** a contributor plans a reproduction, **Then** the claim, metric, source revision, data, baseline, run count, and expected decision rule are recorded before final evaluation.
2. **Given** local resources require a changed dataset, model, or training budget, **When** the attempt is reported, **Then** it is labeled partial or scaled and the changes are listed.
3. **Given** an attempt finishes, **When** a reader checks the report, **Then** paper values and local values are separate and every run, failure, and compute condition is visible.

### Edge Cases

- A learner has no GPU, robot, or access to a model checkpoint.
- A paper is a preprint, changes after the research snapshot, or reports results that cannot be reproduced with this repository's data and compute.
- A simplified example produces different or weak results from a paper.
- A public dataset or checkpoint has access, license, or storage limits.
- A notebook has not been run in the current environment.

## Requirements

### Functional Requirements

- **FR-001**: The project MUST state one learning vision and an ordered path through its current lessons.
- **FR-002**: The project MUST map all current notebooks to a topic, outcome, and evidence boundary.
- **FR-003**: Documentation MUST distinguish paper-reported results from results measured by this project.
- **FR-004**: Each proposed future lesson MUST link primary sources and state its learning outcome, prerequisites, data or model needs, and completion evidence.
- **FR-005**: The roadmap MUST show current gaps and order work by learning value and prerequisites.
- **FR-006**: The core local path MUST NOT require robot hardware or private data.
- **FR-007**: Contributors MUST be able to record intended behavior, design, tasks, and remaining gaps in project specifications.
- **FR-008**: README and curriculum guidance MUST link to the canonical coverage map and research roadmap.
- **FR-009**: Documentation MUST identify the research snapshot date and must not describe a toy example as a paper reproduction.
- **FR-010**: The roadmap MUST identify paper findings that are candidates for local reproduction, with source and data access gates.
- **FR-011**: A reproduction attempt MUST select and record a specific paper claim, metric, baseline, expected decision rule, and run conditions before final evaluation.
- **FR-012**: A reproduction report MUST separate paper-reported and locally measured values and list all deviations and failed runs.
- **FR-013**: A scaled or partial experiment MUST NOT be described as reproducing a full paper result.

### Key Entities

- **Lesson**: A notebook with a topic, learner outcome, prerequisites, data and tool needs, and fidelity status.
- **Research source**: A primary paper or official code source, with date and the specific claim used by a lesson.
- **Roadmap item**: A proposed lesson with priority, outcome, dependencies, resource limits, and acceptance evidence.
- **Evidence claim**: A statement tied either to a source report or to a local run, with its scope and status.
- **Reproduction attempt**: A pre-scoped effort to repeat one reported claim, with source, data, model, protocol, deviations, runs, and status.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All current notebooks appear in the coverage map with a topic and explicit boundary.
- **SC-002**: The README and curriculum guide link to the same canonical coverage map and roadmap.
- **SC-003**: Every roadmap item has at least one primary source, a learner outcome, and completion evidence.
- **SC-004**: A source review finds no documentation claim that says the current notebook reproduces a paper when it does not.
- **SC-005**: Contributors can follow one documented workflow from intended learner outcome to review of remaining work.
- **SC-006**: Unbuilt future lessons remain open in `tasks.md`; documentation work does not mark the full project vision complete.
- **SC-007**: The paper-reproduction matrix lists selected claims, first-party source status, access or resource gates, and an explicit not-run status until local evidence exists.
- **SC-008**: At least one feasible paper claim has a scoped feature specification and local reproduction report, including raw run values, deviations, and a result limited to that experiment.

## Assumptions

- Learners know basic Python, PyTorch, and representation learning.
- Current notebooks are educational examples, not supported production implementations.
- Future paper-aligned work may need external datasets, checkpoints, GPUs, or simulators. The roadmap must state those needs before implementation.
- A reproduction attempt can fail or remain partial because data, checkpoints, compute, or hardware are unavailable. Such a result is useful only when those limits and the exact attempt are recorded.
- The full project vision is a long-term goal. This baseline records and sequences that work; it does not implement every future lesson.
