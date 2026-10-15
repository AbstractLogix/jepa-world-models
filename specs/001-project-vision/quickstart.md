# Quickstart: Spec Kit and Curriculum

## Contributor workflow

This repository was initialized with GitHub Spec Kit 1.2.0 and the Codex skills integration. The shared templates and scripts are under `.specify/`; the generated skills are under `.agents/skills/`.

1. Check the local Spec Kit setup with `uvx --from specify-cli==1.2.0 specify check`.
2. Start a bounded change with `$speckit-specify`.
3. Create its design and tasks with `$speckit-plan` and `$speckit-tasks`.
4. Before implementation, review alignment with `$speckit-analyze`.
5. After implementation, use `$speckit-converge` and keep remaining work open in the task list.

Commit the generated Spec Kit skills when they contain project workflow instructions. Do not store credentials or private tokens under `.agents/`.

## Learner workflow

From the repository root:

```bash
uv sync --locked
uv pip install -r requirements-torch.txt  # optional CUDA 12.8 override; omit for CPU-only use
uv run jupyter lab
```

Start with `notebooks/01_intuition.ipynb`. Use [the coverage map](../../docs/CURRICULUM_COVERAGE.md) to select later lessons and understand their limits. Use [the evaluation protocol](../../docs/EVALUATION_PROTOCOL.md) before comparing results. Use [the roadmap](../../docs/ROADMAP.md) for future topics.

The notebooks were not run as part of this documentation change. Dataset downloads, checkpoint use, and training time depend on the lesson and local hardware.
