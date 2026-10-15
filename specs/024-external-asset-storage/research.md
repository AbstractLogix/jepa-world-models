# Research: External Asset Storage

**Reviewed**: 2026-10-15

## Decisions

### Use a project-level asset root with local fallback

**Decision**: Add `JEPA_ASSETS_DIR` as the parent of the notebook `data/` and `checkpoints/` folders. When unset, retain the repository's current `notebooks/` root.

**Rationale**: The project uses explicit local paths for CIFAR data, Oxford-IIIT Pet data, teaching checkpoints, and the V-JEPA 2.1 checkpoint. A single root covers these paths while preserving the existing clean-clone behavior.

**Alternatives considered**:

- Separate environment variables for every dataset and model: rejected because it adds configuration burden without a current need for independent roots.
- Force all users to store assets outside the repository: rejected because it would break portable local setup on systems without E: or D:.
- Automatically move current assets: rejected because path selection must not move or delete user data.

### Use supported cache environment variables

**Decision**: Document `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, `HF_HUB_CACHE`, `HF_ASSETS_CACHE`, and `TORCH_HOME` for external-drive setup.

**Evidence**: uv documents `UV_CACHE_DIR` as its cache override and `UV_PROJECT_ENVIRONMENT` as its project environment path. It recommends keeping its cache on the same filesystem as the environment where practical. Hugging Face documents `HF_HUB_CACHE` and `HF_ASSETS_CACHE` for model repositories and downstream assets. PyTorch documents `TORCH_HOME` for Torch Hub downloads.

**Rationale**: These variables cover this repository's Python package cache/environment and its current PyTorch and Hugging Face model downloads. `HF_HOME` is not used because it also stores authentication tokens.

**Primary sources**:

- [uv environment variables](https://docs.astral.sh/uv/configuration/environment/)
- [uv project environment path](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)
- [uv cache location and filesystem guidance](https://docs.astral.sh/uv/concepts/cache/#cache-directory)
- [Hugging Face Hub environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
- [PyTorch Hub cache location](https://docs.pytorch.org/docs/2.7/hub.html#where-are-my-downloaded-models-saved)

## Constraints and unverified items

- The external drive must be mounted and the selected asset-root directory must exist before notebook startup.
- The current ignored data, checkpoint, `.venv`, and uv cache directories are not migrated by this feature.
- This feature does not set machine-wide environment variables. Learners must set them in the shell before running `uv sync`, starting Jupyter, or importing the relevant model libraries.
- Storage performance on E: and D: has not been benchmarked. The feature only controls paths.
