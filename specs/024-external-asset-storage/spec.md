# Feature Specification: External Asset Storage

**Feature Branch**: `024-external-asset-storage`

**Created**: 2026-10-15

**Status**: Draft

**Input**: User request: use E: or D: for data files because the system drive is nearly full.

## User Scenarios & Testing

### User Story 1 - Store lesson data and checkpoints outside the checkout (Priority: P1)

A learner with limited space on the system drive selects an external folder for downloaded datasets and model checkpoints. Notebook downloads and saved checkpoints use that folder. Existing local-clone behavior remains available when no external folder is selected.

**Why this priority**: Several lesson assets are large. A learner must be able to place them on another drive before running a lesson.

**Independent Test**: Set an external asset folder, inspect the resolved data and checkpoint paths, and run a small local write/read check. Repeat with the setting absent and confirm that paths resolve under the repository's ignored notebook folders.

**Acceptance Scenarios**:

1. **Given** an existing writable external asset folder, **When** a learner runs a notebook that downloads data or saves a checkpoint, **Then** files are read from or written under that folder.
2. **Given** no external asset folder is selected, **When** a learner runs the same notebook, **Then** it uses the current repository-local data and checkpoint folders.
3. **Given** an external folder is selected but unavailable or unwritable, **When** a notebook needs an asset path, **Then** it stops with the selected path in the error and does not silently write to the repository or system drive.

### User Story 2 - Keep Python and model download caches on the selected drive (Priority: P2)

A learner configures the Python environment, package cache, and framework model caches before the first installation or model download. The setup guide shows commands for WSL and Windows PowerShell.

**Why this priority**: Package and model caches can use several gigabytes. A correct asset folder alone does not keep these caches off a nearly full system drive.

**Independent Test**: Follow each platform's setup guide with paths on a temporary external-drive folder and inspect the resolved environment and cache locations before installing or downloading assets.

**Acceptance Scenarios**:

1. **Given** a learner sets external environment and cache paths before setup, **When** they install project dependencies and optional model assets, **Then** the documented tools use those paths.
2. **Given** the learner follows the default setup on another system, **When** no external-drive path is configured, **Then** the project remains usable with normal platform defaults.

### Edge Cases

- The selected drive is not mounted when a notebook starts.
- The selected path exists but is a file or is not writable.
- A learner has existing assets in the repository and selects a new external folder.
- A saved notebook is opened with a working directory other than the repository root.

## Requirements

### Functional Requirements

- **FR-001**: A learner MUST be able to select one asset root that contains separate data and checkpoint folders.
- **FR-002**: All notebooks that download datasets or read or write model checkpoints MUST resolve those paths through the shared project path configuration.
- **FR-003**: If no asset root is selected, the project MUST retain its current repository-local asset paths.
- **FR-004**: If a selected external root is unavailable or invalid, the path configuration MUST report the selected path and MUST NOT silently fall back to a system-drive path.
- **FR-005**: Setup documentation MUST show how to place the uv environment, uv cache, and supported framework model caches on E: or D: before installing or downloading assets.
- **FR-006**: The project MUST NOT move or delete existing user data or caches as a side effect of selecting a new asset root.
- **FR-007**: Downloaded datasets, checkpoints, and generated outputs MUST remain outside version control.
- **FR-008**: The storage configuration MUST add no runtime dependency.

### Key Entities

- **Asset root**: A user-selected folder that contains the lesson's `data` and `checkpoints` subfolders.
- **Project environment**: The virtual environment used to run the notebooks.
- **Package and model caches**: Downloaded packages or model files stored by uv and supported ML libraries.

## Success Criteria

### Measurable Outcomes

- **SC-001**: With a valid external asset root selected, every in-scope data and checkpoint path resolves beneath that root.
- **SC-002**: With no external asset root selected, every in-scope path resolves to its current repository-local location.
- **SC-003**: An invalid selected root produces an actionable error before any in-scope download or checkpoint write occurs.
- **SC-004**: The WSL and PowerShell setup guides identify the paths for the project environment, uv cache, lesson assets, and framework model caches.
- **SC-005**: The lockfile is unchanged and no package dependency is added for this feature.

## Assumptions

- The learner creates or selects a folder on a mounted drive before running notebooks.
- The external asset root contains `data/` and `checkpoints/`; learners may copy existing files there themselves.
- Local repository folders remain the portable default for users who do not configure an external drive.
- This feature changes future path selection. It does not relocate the current machine's existing uv cache, `.venv`, datasets, or checkpoints.
