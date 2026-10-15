# Quickstart: External Asset Storage

## WSL

Create an asset root on E: or D: and set the variables before installing packages or starting Jupyter. The environment directory is specific to this project; the uv package cache may be shared across projects.

```bash
export UV_CACHE_DIR=/mnt/e/codex-research/uv-cache
export UV_PROJECT_ENVIRONMENT=/mnt/e/codex-research/jepa-world-models-tutorial/.venv
export JEPA_ASSETS_DIR=/mnt/e/codex-research/jepa-world-models-assets
export HF_HUB_CACHE=/mnt/e/codex-research/huggingface/hub
export HF_ASSETS_CACHE=/mnt/e/codex-research/huggingface/assets
export TORCH_HOME=/mnt/e/codex-research/torch
mkdir -p "$JEPA_ASSETS_DIR" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$TORCH_HOME"
uv sync --locked
```

Replace `/mnt/e` with `/mnt/d` to use D:. Set these variables in each new shell before running `uv sync`, starting Jupyter, or importing model libraries.

## Windows PowerShell

```powershell
$env:UV_CACHE_DIR = 'E:\codex-research\uv-cache'
$env:UV_PROJECT_ENVIRONMENT = 'E:\codex-research\jepa-world-models-tutorial\.venv'
$env:JEPA_ASSETS_DIR = 'E:\codex-research\jepa-world-models-assets'
$env:HF_HUB_CACHE = 'E:\codex-research\huggingface\hub'
$env:HF_ASSETS_CACHE = 'E:\codex-research\huggingface\assets'
$env:TORCH_HOME = 'E:\codex-research\torch'
New-Item -ItemType Directory -Force -Path $env:JEPA_ASSETS_DIR, $env:HF_HUB_CACHE, $env:HF_ASSETS_CACHE, $env:TORCH_HOME | Out-Null
uv sync --locked
```

Replace `E:` with `D:` to use D:.

## Check the asset paths

With the project environment active and the variables set, run:

```bash
python -c "from jepa_tutorial.storage import DATA_DIR, CHECKPOINTS_DIR; print(DATA_DIR); print(CHECKPOINTS_DIR)"
```

Both paths must be under the selected `JEPA_ASSETS_DIR`. If the variable is unset, they must resolve under the repository's `notebooks/data` and `notebooks/checkpoints` folders. If the configured root is unavailable, the import must fail and name that path.

The setup commands do not copy or move existing files. Copy any existing data or checkpoints to the new root only when you choose to do so. See the [curriculum coverage map](../../docs/CURRICULUM_COVERAGE.md) for lesson-specific download sizes and requirements.
