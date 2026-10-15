# Implementation Plan: Project Vision and Research Roadmap

**Branch**: `001-project-vision` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Brownfield project baseline for the user request to add Spec Kit, align current lessons, and research future topics.

## Summary

Keep the existing notebooks and make their scope clear. Add a project constitution, a shared evaluation protocol, a source-backed coverage map, a roadmap, and a paper-reproduction matrix. Add the generated Codex Spec Kit integration and use feature specifications, plans, checklists, tasks, and remaining-work records to track work that remains. Use `uv` with a committed lockfile for the base Python environment, while keeping PyTorch's platform-specific wheel install separate. Later features may add runnable lessons and bounded reproduction attempts; this baseline does not claim those results.

## Technical Context

- **Language/version**: Python 3.10 to 3.13 per `pyproject.toml`.
- **Primary dependencies**: PyTorch, torchvision, Gymnasium, JupyterLab, NumPy, SciPy, scikit-learn.
- **Storage**: Local notebooks and source files. Downloaded data and checkpoints stay outside version control.
- **Testing**: No test suite is present in the visible project files. Use `uv lock --check`, an E:-hosted `uv sync --locked` and package-import check, the Spec Kit environment check, and focused document checks for this slice. Notebook execution is not included.
- **Target platform**: Local CPU or GPU notebook environment. GPU is optional for this document change.
- **Project type**: Educational Python repository with Jupyter notebooks.
- **Performance goals**: None for this documentation slice.
- **Constraints**: Do not overwrite staged user work. Keep data, model downloads, and credentials out of the repository. State where external access or hardware is needed.
- **Scale/scope**: Nineteen numbered notebooks; the research roadmap tracks future tutorial modules and paper-aligned reproduction work.

## Constitution Check

- **Teach from evidence**: Pass. The roadmap cites primary papers and separates paper claims from repository results.
- **Label teaching examples honestly**: Pass. The coverage map marks all current lessons and their limits.
- **Make the learning path usable**: Pass for documentation. Current control lessons are identified as Gymnasium CartPole examples.
- **Measure before making performance claims**: Pass. No new performance result is claimed; notebook execution is not reported.
- **Keep changes small and reviewable**: Pass. This slice changes project guidance and adds planning artifacts. It does not rewrite notebook code.
- **PyTorch installation**: The constitution now permits locked CPU wheels and keeps the CUDA wheel override separate from the base environment.

## Project Structure

### Documentation for this feature

```text
specs/001-project-vision/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── tasks.md
├── alignment.md
├── remaining-work.md
└── checklists/requirements.md

docs/
├── CURRICULUM_COVERAGE.md
├── EVALUATION_PROTOCOL.md
├── REPRODUCTION_MATRIX.md
└── ROADMAP.md
```

### Existing source layout

```text
notebooks/                # nineteen learner-facing lessons
src/jepa_tutorial/       # loss, evaluation, and value-shaping helpers
src/planning/            # planning helpers and CartPole demo
src/viz/                 # embedding, value-shaping, and rollout plots
configs/                 # experiment settings
.specify/                # Spec Kit templates, scripts, and project settings
.agents/skills/          # generated Codex Spec Kit skills
```

**Structure Decision**: Keep the existing repository layout. Put learner-facing coverage, evaluation, reproduction, and roadmap documents in `docs/`; put each bounded feature's Spec Kit artifacts under its own generated feature directory.

## Complexity Tracking

No constitution violations or new runtime dependencies are required.
