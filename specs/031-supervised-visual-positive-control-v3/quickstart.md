# Quickstart: Supervised Visual Positive-Control Study v3

This guide describes how to repeat the study after its preregistration and code lock exist. Do not run the training command before the lock is present.

## Environment

Use the existing E: research environment with Python 3.10, CUDA 12.8, and the RTX 5070. Keep the uv cache, virtual environment, transient data, and model checkpoints on E:. The repository already provides the renderer and required packages.

```bash
export UV_CACHE_DIR=/mnt/e/codex-research/jepa-world-models-tutorial/uv-cache
export UV_PROJECT_ENVIRONMENT=/mnt/e/codex-research/jepa-world-models-tutorial/.venv
```

If a new environment is required, sync the locked project dependencies and then apply the CUDA wheel override from `requirements-torch.txt`. Confirm that `torch.cuda.is_available()` is true before the bounded GPU run.

## Validation

From the repository root, freeze the v3 protocol and code hashes once the implementation is complete. Commit the lock before the first fit. The validation command verifies the lock before collecting training and validation data. It does not create reserved test-seed data.

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_visual_jepa_supervised_positive_control_v3.py lock
```

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_visual_jepa_supervised_positive_control_v3.py validation
```

When validation passes, run the gated held-out stage. When it fails, do not run this command.

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_visual_jepa_supervised_positive_control_v3.py heldout
```

## Evidence and figures

Regenerate the figures from committed CSV records and validate their hashes:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/analyze_visual_jepa_supervised_positive_control_v3.py
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/validate_visual_jepa_supervised_positive_control_v3.py --check-integrity
```

The run records, action triplets, validation gates, learning curves, per-coordinate gradient records, figures, and their manifest are the expected outputs. Check the v3 investigation report for measured results and any closed stage.
