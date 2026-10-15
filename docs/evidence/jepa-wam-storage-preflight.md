# JEPA-WAM Storage and Runtime Preflight

**Checked**: 2026-10-15

This task keeps large downloads and runtime files on E: because the user reported low free space on C:. The project checkout stays in its current WSL folder.

## Locations

| Use | Windows path | WSL path | State |
| --- | --- | --- | --- |
| Pinned Python environment | `E:\codex-research\jepa-wam\runtime` | `/mnt/e/codex-research/jepa-wam/runtime` | Python 3.10.12, PyTorch 2.2.0+cu121, NumPy 1.26.4 |
| GPU diagnostic environment | `E:\codex-research\jepa-wam\runtime-blackwell` | `/mnt/e/codex-research/jepa-wam/runtime-blackwell` | Python 3.10.12, PyTorch 2.7.0+cu128, NumPy 1.26.4 |
| pip download cache | `E:\codex-research\jepa-wam\pip-cache` | `/mnt/e/codex-research/jepa-wam/pip-cache` | Used for this task |
| Temporary install files | `E:\codex-research\jepa-wam\tmp` | `/mnt/e/codex-research/jepa-wam/tmp` | Used for this task |
| Model cache destination | `E:\codex-research\jepa-wam\huggingface` | `/mnt/e/codex-research/jepa-wam/huggingface` | Empty; no model files downloaded |
| Torch and CUDA caches | `E:\codex-research\jepa-wam\torch` | `/mnt/e/codex-research/jepa-wam/torch` | Used for this task |
| Benchmark asset destination | `E:\codex-research\jepa-wam\assets` | `/mnt/e/codex-research/jepa-wam/assets` | Empty; no asset archive downloaded |
| Pinned source checkouts | `E:\codex-research\jepa-wam\source` | `/mnt/e/codex-research/jepa-wam/source` | JEPA-WAM and LIBERO-Plus source only |
| Runtime logs and evidence | `E:\codex-research\jepa-wam\evidence` | `/mnt/e/codex-research/jepa-wam/evidence` | Kept outside the Git project |
| Project checkout | Not moved | `/home/oscar/projects/education/jepa-world-models-tutorial` | Existing code and documents remain here |

## Other disk use

- A stopped, partial environment remains at `/tmp/codex-jepa-wam-py310` on the WSL system disk. It is about 19 MB. It was not removed.
- The existing shared pip cache at `/home/oscar/.cache/pip` is about 4.9 GB. It was left untouched.
- At this check, E: had 503 GB free and D: had 578 GB free. No WAM model checkpoint, model weights, simulator asset archive, or RLDS training data were downloaded.

## GPU compatibility

- The pinned PyTorch 2.2.0+cu121 environment can see the RTX 5070, but its CUDA matrix operation fails with `no kernel image is available`. This wheel does not include the GPU's `sm_120` architecture.
- The separate PyTorch 2.7.0+cu128 diagnostic environment passes a CUDA matrix operation on the RTX 5070. This only checks basic CUDA support. It does not confirm that JEPA-WAM imports, model loading, rendering, or evaluation work.
- The WAM one-task diagnostic remains open. No model was loaded and no simulation ran.
