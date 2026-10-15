# Quickstart: Visual JEPA Decision-Usefulness Lesson

## Requirements

- Python 3.10–3.13 and `uv`.
- A local CPU for the simulator and data generation.
- Optional NVIDIA GPU with a compatible CUDA PyTorch build. The notebook checks the build with a synchronized tensor operation and falls back to CPU when the check fails.
- No downloaded dataset, model checkpoint, or robot hardware.

Keep the environment, uv cache, model caches, and larger run outputs on E: or D: if the system drive is low on space. For WSL on E:, set paths before setup:

```bash
export UV_CACHE_DIR=/mnt/e/codex-research/jepa-world-models-tutorial/uv-cache
export UV_PROJECT_ENVIRONMENT=/mnt/e/codex-research/jepa-world-models-tutorial/.venv
export JEPA_ASSETS_DIR=/mnt/e/codex-research/jepa-world-models-assets
mkdir -p "$UV_CACHE_DIR" "$JEPA_ASSETS_DIR"
uv sync --locked
```

For CPU-only use, skip the CUDA command. For CUDA 12.8, install the project's existing GPU wheels after sync:

```bash
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -r requirements-torch.txt
```

Check the selected runtime:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

## Run the lesson

Start in validation-only pilot mode. This trains one development seed on the fixed 16/4 episode split, writes a pilot record with a training-code hash, and does not collect final intervention or episode cases.

```bash
export JEPA_RUN_MODE=pilot
"$UV_PROJECT_ENVIRONMENT/bin/python" -m jupyter lab
```

Open `notebooks/22_visual_jepa_decision_usefulness.ipynb` and run all cells from a fresh kernel. Read the collapse status in the run report. If the pilot passes, restart the kernel with `JEPA_RUN_MODE=full` and run the notebook again. Full mode checks that the pilot hash, objective, split, seeds, and device match. If the pilot fails or is missing, full mode skips final model training and evaluation.

For a short pipeline check, use `JEPA_RUN_MODE=smoke`. Smoke mode uses separate reserved seeds and does not satisfy the research comparison. To force CPU on a host with CUDA, set `CUDA_VISIBLE_DEVICES=""` before launching Jupyter.

The pilot model row and larger outputs should stay on E: or D:. The validation-pilot CSV is a small audit record in `docs/evidence/`. Final model, intervention, and episode records are written only after their gates pass; generated training arrays and model weights are not committed.

## Expected checks

- The notebook prints the selected CPU/CUDA device, PyTorch and CUDA versions, precision, elapsed time, and peak GPU memory when available.
- CSV validation confirms that action rows remain grouped by held-out start and that methods share evaluation starts.
- Final model training and evaluation remain closed until a protocol-hash-matched validation pilot passes. The five-seed result is a **comparison** only if every trained method completes all five reserved model seeds and the required paired episodes. Otherwise the report says **exploratory** or **partial** and lists missing runs.
- Results apply only to this synthetic MountainCar task. They do not reproduce V-JEPA 2, LeJEPA, V-JEPA 2.1, or RoboJEPA.
