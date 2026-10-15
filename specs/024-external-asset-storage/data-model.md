# Data Model: External Asset Storage

## Asset root

Represents the selected parent folder for notebook assets.

| Field | Type | Rule |
| --- | --- | --- |
| configured path | filesystem path or unset | If set, it must exist, be a directory, and be writable. |
| data path | filesystem path | `<asset root>/data`; used for downloaded datasets. |
| checkpoint path | filesystem path | `<asset root>/checkpoints`; used for downloaded or generated model checkpoints. |

When the configured path is unset, the asset root is the repository's `notebooks/` directory.

## Environment and download caches

These are user configuration values, not project-managed data entities.

| Setting | Purpose | Default behavior if unset |
| --- | --- | --- |
| `UV_PROJECT_ENVIRONMENT` | Project virtual environment | uv uses `.venv` in the project. |
| `UV_CACHE_DIR` | Package download/build cache | uv uses its platform cache directory. |
| `HF_HUB_CACHE` | Hugging Face model and dataset repository cache | Hugging Face uses its platform cache under `HF_HOME` or the home cache. |
| `HF_ASSETS_CACHE` | Hugging Face downstream asset cache | Hugging Face uses its platform cache under `HF_HOME` or the home cache. |
| `TORCH_HOME` | PyTorch Hub model cache | PyTorch uses its platform cache under the home cache. |

Selecting new paths does not move existing files or credentials.
