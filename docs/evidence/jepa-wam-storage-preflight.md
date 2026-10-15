# JEPA-WAM Storage, Runtime, and Diagnostic Evidence

**Checked**: 2026-10-15

This task keeps large downloads and runtime files on E: because the user reported low free space on C:. The project checkout stays in its current WSL folder.

## Locations

| Use | Windows path | WSL path | State |
| --- | --- | --- | --- |
| Pinned Python environment | `E:\codex-research\jepa-wam\runtime` | `/mnt/e/codex-research/jepa-wam/runtime` | Python 3.10.12, PyTorch 2.2.0+cu121, NumPy 1.26.4 |
| GPU diagnostic environment | `E:\codex-research\jepa-wam\runtime-blackwell` | `/mnt/e/codex-research/jepa-wam/runtime-blackwell` | Python 3.10.12, PyTorch 2.7.0+cu128, NumPy 1.26.4 |
| Wand Python overlay | `E:\codex-research\jepa-wam\python-overlays\wand` | `/mnt/e/codex-research/jepa-wam/python-overlays/wand` | Wand 0.7.2, scikit-image 0.25.2, lazy-loader 0.4 |
| MagickWand native library | `E:\codex-research\jepa-wam\native-magick\root\usr` | `/mnt/e/codex-research/jepa-wam/native-magick/root/usr` | Local copy; used by the completed task-render diagnostic |
| Project uv environment | `E:\codex-research\jepa-world-models-tutorial\.venv` | `/mnt/e/codex-research/jepa-world-models-tutorial/.venv` | Python 3.13.14, PyTorch 2.10.0+cu128; CUDA matrix check passed |
| uv package cache | `E:\codex-research\uv-cache` | `/mnt/e/codex-research/uv-cache` | Used for the project environment |
| pip download cache | `E:\codex-research\jepa-wam\pip-cache` | `/mnt/e/codex-research/jepa-wam/pip-cache` | Used for this task |
| Temporary install files | `E:\codex-research\jepa-wam\tmp` | `/mnt/e/codex-research/jepa-wam/tmp` | Used for this task |
| Hugging Face cache and logs | `E:\codex-research\jepa-wam\huggingface` | `/mnt/e/codex-research/jepa-wam/huggingface` | Cache and transfer logs stay on E:; model payloads use the asset paths below |
| Torch and CUDA caches | `E:\codex-research\jepa-wam\torch` | `/mnt/e/codex-research/jepa-wam/torch` | Used for this task |
| Benchmark asset destination | `E:\codex-research\jepa-wam\assets` | `/mnt/e/codex-research/jepa-wam/assets` | Required model files and metadata are complete and verified; LIBERO-Plus archive is extracted |
| Pinned source checkouts | `E:\codex-research\jepa-wam\source` | `/mnt/e/codex-research/jepa-wam/source` | JEPA-WAM and LIBERO-Plus source only |
| Runtime logs and evidence | `E:\codex-research\jepa-wam\evidence` | `/mnt/e/codex-research/jepa-wam/evidence` | Kept outside the Git project |
| Project checkout | Not moved | `/home/oscar/projects/education/jepa-world-models-tutorial` | Existing code and documents remain here |

## Other disk use

- A stopped, partial environment remains at `/tmp/codex-jepa-wam-py310` on the WSL system disk. It is about 19 MB. It was not removed.
- The existing shared pip cache at `/home/oscar/.cache/pip` is about 4.9 GB. It was left untouched.
- The pre-existing project `.venv` is about 405 MB and remains in the checkout until the E:-hosted uv environment passes its import check. New uv packages and temporary install files use E:.
- E: has 464 GB free at the latest check. The policy checkpoint, base VLM, Qwen2.5-0.5B weights, V-JEPA checkpoint, and LIBERO-Plus archive are complete. The three JSON run metadata files match their pinned Hub Git blob IDs and sizes. The V-JEPA checkpoint's local SHA-256 is recorded in the reproduction report; the provider publishes no SHA-256. The LIBERO-Plus archive passed path checks and extracted to the staged source tree. No RLDS training data was downloaded.

## GPU compatibility

- The pinned PyTorch 2.2.0+cu121 environment can see the RTX 5070, but its CUDA matrix operation fails with `no kernel image is available`. This wheel does not include the GPU's `sm_120` architecture.
- The separate Python 3.10.12 and PyTorch 2.7.0+cu128 diagnostic environment passes a CUDA matrix operation on the RTX 5070. The pinned source and evaluator import, and an isolated EGL context check passes. These checks do not confirm a LIBERO task render, model loading, or evaluation.
- The project uv environment on E: uses Python 3.13.14 and PyTorch 2.10.0+cu128. A CUDA matrix operation passes on the RTX 5070. This checks the project wheel and device path, not the WAM evaluator or benchmark.
- The official evaluator imports from the pinned checkout with the E:-hosted Python overlay. TensorFlow 2.15.0 and TFDS 4.9.3 import; `dlimp` is pinned to `040105d256bd28866cc6620621a3d5f7b6b91b46`. TensorFlow does not register a GPU, but PyTorch loaded the model on the RTX 5070.
- The one-task diagnostic completed on 2026-10-15. It rendered the first spatial task, took ten no-op warm-up steps and one policy action, and reported 0/1 task success. The episode took 53.64 seconds. Sampled GPU memory reached about 6.6 GiB; a peak-memory measurement was not captured. This is pipeline evidence, not a paper result. See [the reproduction report](../reproductions/jepa-wam-libero-plus.md).
