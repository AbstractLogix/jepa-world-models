# JEPA-WAM LIBERO-Plus Checkpoint Re-evaluation

**Reviewed**: 2026-10-15 | **Status**: Environment preflight; no model or simulator assets were downloaded. No model was loaded and no simulation ran.

## Target claim

The target is the 79.2% average success rate reported for JEPA-WAM on LIBERO-Plus Table 2 of [arXiv:2608.09381](https://arxiv.org/html/2608.09381). The paper reports results across seven perturbation categories. This project would re-evaluate the released checkpoint. It would not repeat the paper's training.

The [author repository at commit 8bfe3b31a79da57fb217fe10d0e1dc3b282b1c07](https://github.com/SpriteWithoutIce/JEPA_WAM/tree/8bfe3b31a79da57fb217fe10d0e1dc3b282b1c07) gives the command scope as the LIBERO-Plus spatial suite, all categories, one trial per task, and reports the same 79.2% result. The official [LIBERO-Plus repository](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182) says one trial per task is its standard protocol and describes 10,030 perturbed tasks across four suites. Its pinned task classification lists 2,402 spatial-suite variants. The full spatial command is therefore a 2,402-episode run, not a ten-episode check.

The paper says to follow each benchmark's standard evaluation protocol but does not state a trial count in its Appendix B. The author README gives a concrete current command and result mapping. Record this as the operational release-checkpoint target; do not imply that every paper detail is independently specified.

## Training and checkpoint mismatch

The paper reports eight-GPU LIBERO training with global batch size 128 for 60,000 steps. The pinned release README describes an eight-GPU recipe with global batch size 256 for 40,000 steps. Its released policy file is named step-040000-epoch-37-loss=0.0262.pt. These differences mean that a released-checkpoint result can validate the evaluation path and re-check the published score, but cannot reproduce the paper's training run.

The repository and checkpoint are licensed MIT. The required Qwen2.5-0.5B files are Apache-2.0. The official V-JEPA 2 repository states MIT terms for its V-JEPA 2.1 code and models. The LIBERO-Plus dataset card states MIT. These are separate third-party assets; keep their individual terms with the transfer record.

## Pinned release artifacts

The checkpoint repository is [CokeAnd1ce/JEPA_WAM at revision ca10ccbc191d8f56b4346487913e043b2722b6d2](https://huggingface.co/CokeAnd1ce/JEPA_WAM/tree/ca10ccbc191d8f56b4346487913e043b2722b6d2). Its whole repository is reported as 373 GB, so do not clone or download the whole repository.

| Required for released-checkpoint evaluation | Published size | Hub file object ID |
| --- | ---: | --- |
| LIBERO policy checkpoint, step 40,000 | 5,355,388,110 bytes | 5e71320d3c304e227e8420c763a5866da30f5cfa |
| Base VLM checkpoint | 2,525,831,255 bytes | 315125023c004f43ad1d6e5042fae47e9db885f2 |
| Qwen2.5-0.5B weights | 988 MB | Recheck at transfer |
| V-JEPA 2.1 ViT-L/16 384px checkpoint | Not recorded here | Recheck at transfer |
| LIBERO-Plus simulator assets archive | 6.4 GB | Recheck at transfer |

The Hub object IDs above are not recorded as SHA-256 checks. Verify file hashes after transfer. The evaluation launcher needs the policy checkpoint, base VLM directory, Qwen directory, V-JEPA checkpoint, and LIBERO-Plus checkout with its extended assets. It does not need the LIBERO RLDS training data. Do not transfer training data for this evaluation.

At the pinned code revision, V-JEPA checkpoint loading calls torch.load with weights_only=False. Keep any later load inside a disposable, isolated environment and use only the pinned first-party artifact.

## Host preflight

- The host has an RTX 5070 with 12,227 MiB VRAM.
- The pinned diagnostic environment is under `E:\codex-research\jepa-wam\runtime` (WSL: `/mnt/e/codex-research/jepa-wam/runtime`). It has Python 3.10.12, PyTorch 2.2.0+cu121, and NumPy 1.26.4. PyTorch sees the GPU, but a CUDA matrix operation fails because this wheel has no kernel image for the GPU's `sm_120` architecture.
- A separate diagnostic environment is under `E:\codex-research\jepa-wam\runtime-blackwell`. It has PyTorch 2.7.0+cu128 and passes a CUDA matrix operation on this RTX 5070. This environment differs from the pinned release environment and does not establish benchmark compatibility.
- Pinned JEPA-WAM and LIBERO-Plus source checkouts are under `E:\codex-research\jepa-wam\source`. The task-specific package cache, temporary files, model-cache destination, asset destination, and evidence destination are also on E:. No checkpoint, model weights, LIBERO-Plus asset archive, or RLDS training data have been downloaded.
- E: had 503 GB free and D: had 578 GB free at the storage check. The project checkout remains at `/home/oscar/projects/education/jepa-world-models-tutorial`; it was not moved. A small abandoned environment remains at `/tmp/codex-jepa-wam-py310`, and the existing shared pip cache at `/home/oscar/.cache/pip` was left untouched. See [storage preflight evidence](../evidence/jepa-wam-storage-preflight.md) for the full location list.
- The import preflight reaches the pinned LIBERO code but stops because the Python package `wand` is missing. The host also has no discoverable MagickWand shared library. No system packages were added. MuJoCo/EGL rendering has not been checked. No model was loaded and no simulation ran.

## Bounded next steps

1. In a disposable checkout and environment, verify the exact inference artifacts, their file hashes, and third-party terms. Do not download the 373 GB checkpoint repository or RLDS training data.
2. Run the upstream one-task, one-action diagnostic with one task and one episode step. Record dependency versions, CUDA, GPU memory, renderer status, elapsed time, and any load errors. This is a pipeline smoke only, not a success-rate result.
3. Only if the smoke passes, estimate the full spatial-suite runtime. The paper-aligned checkpoint re-evaluation is 2,402 task variants, all seven categories, one trial per task, with no task or episode-step caps.
4. If the full run completes, preserve all 2,402 task outcomes and report category rates, the overall average, runtime, peak GPU memory, checkpoint and source revisions, and deviations. Compare the local values to the paper and author README. Keep the training-step and batch-size mismatch explicit.

The one-action diagnostic remains open. The 2,402-task run remains pending. No JEPA-WAM result is currently local.
