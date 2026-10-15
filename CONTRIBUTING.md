# Contributing

Thank you for helping improve this course. Contributions can improve a lesson, add evidence for a research claim, or make the learning path easier to use.

## Choose a starting point

- **Learner or instructor:** start with the [curriculum guide](CURRICULUM_GUIDE.md).
- **Research contributor:** read the [evaluation protocol](docs/EVALUATION_PROTOCOL.md) and [reproduction matrix](docs/REPRODUCTION_MATRIX.md).
- **New lesson or substantial change:** use the [Spec Kit quickstart](specs/001-project-vision/quickstart.md).

## Before adding a lesson

Record the learner outcome, prerequisites, primary sources, required data and models, expected output, and local validation path. For a research reproduction, also select one exact claim and record its metric, baseline, data split, run count, and decision rule before the final evaluation.

Keep paper-reported results separate from local measurements. Label a toy example as **Illustrative**. A reduced or changed setup is a partial or scaled attempt, not an exact reproduction. Do not commit datasets, checkpoints, credentials, or generated run outputs.

## Spec Kit workflow

For a new lesson or substantial project change, create a bounded feature and follow this order:

1. `$speckit-specify` — state user needs and acceptance checks.
2. `$speckit-plan` — record design choices and limits.
3. `$speckit-tasks` — create ordered implementation tasks.
4. `$speckit-analyze` — check the design artifacts for consistency.
5. `$speckit-implement` — complete the planned work.
6. `$speckit-converge` — record any remaining implementation gaps.

The project templates and scripts are under `.specify/`. Codex skills are under `.agents/skills/`. Keep the feature specification, curriculum coverage, reproduction report, and roadmap in sync when a lesson changes.

## Checks before a pull request

- For documentation changes, check links and confirm the Markdown renders as intended.
- For notebook code changes, run the changed notebook from a clean kernel when the needed data and device are available. Record the Python and library versions, device, command, and result.
- For a measured comparison, preserve raw per-run values and list failed or missing runs. Follow the [evaluation protocol](docs/EVALUATION_PROTOCOL.md).
- State which checks you ran and which remain open. Do not describe an unrun check as passing.

Use short, direct language. Define a technical term when learners first need it. Keep download size, license, hardware needs, and result limits visible before a learner starts a run.
