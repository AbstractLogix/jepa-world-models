# Quickstart: JEPA Action Fidelity v2

The v2 protocol and its amendments must be locked before any model fit. The runner writes all transient data and checkpoints to E: or D:. It does not change Notebook 22, Notebook 23, or their evidence.

## Set up on E:

From the repository root:

~~~bash
export UV_PROJECT_ENVIRONMENT=/mnt/e/codex-research/jepa-world-models-tutorial/.venv
export JEPA_V2_OUTPUT_DIR=/mnt/e/codex-research/jepa-world-models-tutorial/visual-jepa-action-fidelity-v2
uv sync --locked
uv run --locked --no-sync python -m unittest discover -s tests -p test_visual_jepa_action_fidelity_v2.py
~~~

## Freeze and run the pipeline

Create the implementation and protocol lock first. Then run the retained v1 checkpoint audit and the reserved CPU smoke. Smoke fits check the pipeline only; they do not count as validation or evaluation.

~~~bash
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py lock --confirm-pretraining
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py verify-lock
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py audit-v1 --device cuda --output-dir "$JEPA_V2_OUTPUT_DIR/v1-audit"
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py smoke --output-dir "$JEPA_V2_OUTPUT_DIR/smoke"
~~~

The GPU validation run uses the three preregistered paired model/data seeds and stops at the fixed compute, memory, or storage limits. The runner opens held-out evaluation only when the validation gates pass. It opens planning only when the held-out action and rollout gates pass.

~~~bash
uv run --locked --no-sync python scripts/run_visual_jepa_action_fidelity_v2.py validate --device cuda --output-dir "$JEPA_V2_OUTPUT_DIR/validation"
uv run --locked --no-sync python scripts/validate_visual_jepa_action_fidelity_v2.py --run-dir "$JEPA_V2_OUTPUT_DIR/validation" --v1-audit-dir "$JEPA_V2_OUTPUT_DIR/v1-audit" --retain
~~~

If validation fails a gate, the runner records the reason and keeps later stages closed. Do not change the code or protocol after locking. A required repair needs a new dated amendment and lock before another validation fit.

## Regenerate figures

Figures use only retained raw records. They do not train or evaluate models.

~~~bash
uv run --locked --no-sync python scripts/audit_visual_jepa_action_fidelity_v2_observations.py
uv run --locked --no-sync python scripts/analyze_visual_jepa_action_fidelity_v2.py --output-dir docs/portfolio/figures/visual-jepa-action-fidelity-v2
uv run --locked --no-sync python scripts/analyze_visual_jepa_action_fidelity_v2.py --output-dir docs/portfolio/figures/visual-jepa-action-fidelity-v2 --verify-determinism
uv run --locked --no-sync python scripts/validate_visual_jepa_action_fidelity_v2.py --check-integrity
uv run --locked --no-sync python scripts/audit_visual_jepa_action_fidelity_v2_observations.py --check-integrity
~~~

The full study is bounded to 90 GPU-minutes, 2 GiB allocated GPU memory, and 5 GiB of transient output. The conditional planning stage stops at 750 million model transition calls. No Push-T or JEPA-WAM LIBERO-Plus benchmark is part of this protocol.
