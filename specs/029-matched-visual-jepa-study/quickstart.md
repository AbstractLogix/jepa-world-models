# Quickstart: Matched Visual JEPA Decision Study v1

## Requirements

- Python 3.10–3.13 and the locked `uv` environment.
- CPU for unit checks and smoke mode.
- Optional NVIDIA CUDA environment for model training and full visual-model planning.
- No dataset, pretrained checkpoint, or robot hardware.

Keep the environment and run output on E: or D: when C: has limited space. The prepared Windows/WSL environment uses E:. Example:

```bash
export UV_PROJECT_ENVIRONMENT=/mnt/e/codex-research/jepa-world-models-tutorial/.venv
export JEPA_RUN_OUTPUT_DIR=/mnt/e/codex-research/jepa-world-models-tutorial/run-output/visual-jepa-decision-study-v1
mkdir -p "$JEPA_RUN_OUTPUT_DIR"
```

Use the standard project setup for a fresh checkout. For the optional CUDA 12.8 runtime, follow the existing [GPU setup instructions](../../README.md) and confirm that `torch.cuda.is_available()` is true before full mode.

## Run checks and CPU smoke

```bash
uv run --locked --no-sync python -m unittest discover -s tests -p 'test_decision_study.py' -v
uv run --locked --no-sync python scripts/run_visual_jepa_decision_study.py smoke --output-dir "$JEPA_RUN_OUTPUT_DIR" --device cpu
```

Smoke seeds are separate from development, training, intervention, and final evaluation. Smoke mode cannot open the final gate.

## Run the study

Run these stages in order. Do not run `full` if an earlier gate fails.

```bash
uv run --locked --no-sync python scripts/run_visual_jepa_decision_study.py dev-search --output-dir "$JEPA_RUN_OUTPUT_DIR" --device cuda
uv run --locked --no-sync python scripts/run_visual_jepa_decision_study.py oracle-gate --output-dir "$JEPA_RUN_OUTPUT_DIR" --device cpu
uv run --locked --no-sync python scripts/run_visual_jepa_decision_study.py full --output-dir "$JEPA_RUN_OUTPUT_DIR" --device cuda
```

The first stage uses only declared development seeds and writes the protocol lock. The second uses final starts for exact-dynamics planning only. It writes a passing gate only at 18/20 successes or higher. The full stage refuses a missing or failing gate.

## Validate evidence and rebuild figures

```bash
uv run --locked --no-sync python scripts/run_visual_jepa_decision_study.py validate --output-dir "$JEPA_RUN_OUTPUT_DIR"
uv run --locked --no-sync python scripts/analyze_visual_jepa_decision_study.py
uv run --locked --no-sync python scripts/generate_visual_jepa_decision_study_figures.py
```

The analysis command reports the primary paired return estimate. It resamples training-data/model seed pairs and then resamples the shared evaluation starts once per bootstrap replicate. The figure command reads the retained v1 evidence and historical Notebook 22 episode evidence. It builds one A–D dashboard plus a separate E architecture diagram. It fails if the required evidence rows are missing. Neither command trains models or downloads data.

Copy only the small CSV/JSON audit records and deterministic figures into the repository. Keep temporary arrays, weights, and caches outside the checkout. Read the versioned [study report](../../docs/reproductions/visual-jepa-decision-study-v1.md) before using any result.

## Evidence limits

The task uses synthetic frames and a small locally trained model. The exact and learned state-space methods see privileged state. A visual state probe is trained from task labels and supports a common task score; it is a diagnostic adapter, not proof that a latent planner is useful. A failed oracle gate blocks claims about learned-model failure.
