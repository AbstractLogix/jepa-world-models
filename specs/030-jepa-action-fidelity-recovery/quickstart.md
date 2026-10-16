# Quickstart: JEPA Action Fidelity v2

This protocol is frozen before model training. Run outputs belong on E: or D:. Do not run the held-out or planning stage until the validator reports that its gate is open.

## CPU setup and smoke

From the repository root:

~~~bash
uv sync --locked
uv run --locked --no-sync python -m unittest tests.test_visual_jepa_action_fidelity_v2
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py smoke --device cpu --output-dir /tmp/visual-jepa-action-fidelity-v2-smoke
~~~

Smoke uses reserved seeds only. It does not run the full validation fit or open held-out cases.

## CUDA validation

First verify the E:-hosted Python, PyTorch, CUDA, and GPU. Then set the output path on E: and run the locked validation stage:

~~~bash
export UV_PROJECT_ENVIRONMENT=/mnt/e/codex-research/jepa-world-models-tutorial/.venv
export JEPA_V2_OUTPUT_DIR=/mnt/e/codex-research/jepa-world-models-tutorial/action-fidelity-v2
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py validate --device cuda --output-dir "$JEPA_V2_OUTPUT_DIR"
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py validate-evidence --output-dir "$JEPA_V2_OUTPUT_DIR"
~~~

The runner writes checkpoints and transient data to the supplied E: path. It writes compact evidence to the repository only after schema and hash checks pass.

## Held-out and planning gates

The following commands must refuse to run unless the prior stage passed and its protocol/code hashes match:

~~~bash
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py heldout --device cuda --output-dir "$JEPA_V2_OUTPUT_DIR"
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py planning --device cuda --output-dir "$JEPA_V2_OUTPUT_DIR"
~~~

Do not run these commands when a gate fails. Preserve the validation record and report the negative or inconclusive result.

## Reports and figures

~~~bash
uv run --locked --no-sync python scripts/analyze_visual_jepa_action_fidelity_v2.py --evidence-dir docs/evidence
uv run --locked --no-sync python scripts/generate_visual_jepa_action_fidelity_v2_figures.py --evidence-dir docs/evidence
~~~

Figures regenerate from retained v2 CSV/JSON records. No command above trains or evaluates a model during figure generation.
