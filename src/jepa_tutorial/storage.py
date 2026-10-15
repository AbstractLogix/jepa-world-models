"""Resolve shared dataset and checkpoint locations for the tutorial notebooks.

Set ``JEPA_ASSETS_DIR`` to an existing writable folder to store notebook
assets outside the repository. Without it, assets stay under ``notebooks/``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_configured_assets_dir = os.environ.get("JEPA_ASSETS_DIR")

if _configured_assets_dir is None:
    ASSETS_DIR = REPOSITORY_ROOT / "notebooks"
else:
    if not _configured_assets_dir:
        raise ValueError("JEPA_ASSETS_DIR is set but empty; select an existing writable directory.")

    _configured_path = Path(_configured_assets_dir).expanduser()
    if not _configured_path.is_absolute():
        _configured_path = REPOSITORY_ROOT / _configured_path

    try:
        ASSETS_DIR = _configured_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"JEPA_ASSETS_DIR '{_configured_assets_dir}' must exist and be accessible. "
            "Create or mount the selected directory, then restart the notebook kernel."
        ) from error

    if not ASSETS_DIR.is_dir():
        raise ValueError(
            f"JEPA_ASSETS_DIR '{_configured_assets_dir}' is not a directory. "
            "Select an existing writable directory."
        )
    if not os.access(ASSETS_DIR, os.W_OK | os.X_OK):
        raise ValueError(
            f"JEPA_ASSETS_DIR '{_configured_assets_dir}' is not writable. "
            "Select a directory where this account can create files."
        )

DATA_DIR = ASSETS_DIR / "data"
CHECKPOINTS_DIR = ASSETS_DIR / "checkpoints"

__all__ = ["REPOSITORY_ROOT", "ASSETS_DIR", "DATA_DIR", "CHECKPOINTS_DIR"]
