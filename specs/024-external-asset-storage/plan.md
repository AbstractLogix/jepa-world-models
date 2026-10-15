# Implementation Plan: External Asset Storage

**Branch**: `024-external-asset-storage` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/024-external-asset-storage/spec.md`

## Summary

Add one shared path resolver for notebook datasets and checkpoints. Keep the current repository-local folders as the default. Let learners redirect those assets and the uv, Hugging Face Hub, and Torch caches to an existing E: or D: folder before setup. Do not move or delete current assets or caches.

## Technical Context

**Language/Version**: Python 3.10–3.13

**Primary Dependencies**: Standard library only for path resolution; existing uv-managed notebook dependencies remain unchanged.

**Storage**: Filesystem directories for ignored data, checkpoints, virtual environments, and download caches.

**Testing**: Focused path-resolution checks, notebook JSON/nbformat validation, and documentation path review. Do not install packages or download assets for validation.

**Target Platform**: WSL/Linux and Windows PowerShell; local CPU and GPU notebook environments.

**Project Type**: Python package with educational Jupyter notebooks and documentation.

**Performance Goals**: No added work during notebook startup beyond resolving configured paths.

**Constraints**: Preserve `uv.lock` and current local defaults. An explicitly selected but missing or invalid asset root must fail before any notebook download or checkpoint write. Keep data and checkpoints out of version control. Leave the existing `.venv`, uv cache, datasets, and checkpoints in place.

**Scale/Scope**: One shared path module and the eight notebooks that download datasets or read/write checkpoints: 02, 03, 04, 05, 06, 09, 11, and 13. Update learner setup and coverage documentation.

## Constitution Check

- **Teach from evidence**: Pass. The change makes no model or paper-performance claims.
- **Label teaching examples honestly**: Pass. Notebook results and labels remain unchanged.
- **Make the learning path usable**: Pass. Learners can keep optional assets off the system drive while local defaults remain available.
- **Measure before making performance claims**: Pass. This feature changes paths only.
- **Keep changes small and reviewable**: Pass. Use one path module, no new dependency, and a small set of notebook path substitutions.
- **Data and credentials**: Pass. Downloads and caches remain outside version control. Use `HF_HUB_CACHE`, not `HF_HOME`, so model-cache relocation does not relocate Hugging Face credentials.

## Project Structure

### Documentation

```text
specs/024-external-asset-storage/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/storage-paths.md
├── checklists/requirements.md
└── tasks.md
```

### Source and tutorials

```text
src/jepa_tutorial/storage.py
notebooks/02_lejepa_from_scratch.ipynb
notebooks/03_rectified_lpjepa.ipynb
notebooks/04_vjepa_plus_v1_jepa.ipynb
notebooks/05_value_guided_planning.ipynb
notebooks/06_gradient_based_planning.ipynb
notebooks/09_vjepa21_dense_features.ipynb
notebooks/11_vl_jepa.ipynb
notebooks/13_object_centric_lejepa.ipynb
README.md
CURRICULUM_GUIDE.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
```

**Structure Decision**: Put the path resolver in the existing `jepa_tutorial` package. Notebook code imports its data and checkpoint roots. Setup documentation sets external-drive environment variables before uv, Jupyter, or model libraries start.

## Design

The resolver uses `JEPA_ASSETS_DIR` as a parent folder with `data/` and `checkpoints/` children. If unset, it resolves to the repository's existing `notebooks/` folder. If set, the root must already exist, be a directory, and be writable; failure is explicit and does not fall back to the repository or system drive. Importing the module does not create directories.

Keep the uv project environment path project-specific. Place the uv cache on the same external volume when practical. Set `HF_HUB_CACHE`, `HF_ASSETS_CACHE`, and `TORCH_HOME` before starting Python so optional model downloads also stay on that volume. Do not use `HF_HOME`, because it also stores Hugging Face credentials.

No compatibility symlinks, data migration, cache cleanup, or lockfile edits are in scope.

## Validation

1. Resolve default paths with `JEPA_ASSETS_DIR` unset and confirm they point to `notebooks/data` and `notebooks/checkpoints` in the repository.
2. Resolve paths under a temporary existing external root and perform a small write/read check under both child folders.
3. Set the variable to a missing path and verify import fails with the selected path and no fallback path is created.
4. Validate all eight affected notebooks with nbformat and inspect that each asset read/write uses the shared resolver.
5. Confirm `uv.lock` is unchanged, `git diff --check` passes, and no downloaded assets or cache files are added to Git.
