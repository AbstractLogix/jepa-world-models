# Storage Path Contract

## Learner configuration

`JEPA_ASSETS_DIR` is an optional environment variable. It names an existing writable directory that contains or will contain two child directories:

- `data/` for downloaded datasets.
- `checkpoints/` for downloaded and generated model files.

The variable must be set before the notebook kernel starts. Use a WSL path such as `/mnt/e/research/jepa-assets` in WSL and a Windows path such as `E:\research\jepa-assets` in native PowerShell.

## Resolution

| Configuration | Resolved root |
| --- | --- |
| `JEPA_ASSETS_DIR` is unset | `<repository>/notebooks` |
| `JEPA_ASSETS_DIR` is an existing writable directory | The configured directory |
| Configured directory is absent, not a directory, or not writable | Raise an actionable error containing the configured path; do not fall back |

The project does not create the configured root on import. Notebook download or checkpoint code may create its `data/` or `checkpoints/` child when needed.

## Cache configuration

Set the supported `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, `HF_HUB_CACHE`, `HF_ASSETS_CACHE`, and `TORCH_HOME` variables before setup or model imports when those caches must use an external drive. These variables do not move existing cache contents.
