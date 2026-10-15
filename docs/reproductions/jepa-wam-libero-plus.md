# JEPA-WAM LIBERO-Plus Checkpoint Re-evaluation

**Reviewed**: 2026-10-15 | **Status**: The bounded one-task diagnostic passed. The pinned model loaded on the RTX 5070, the evaluator rendered one LIBERO spatial task, and it took ten no-op warm-up steps followed by one policy action. The task result was 0/1, so this is pipeline evidence, not a success-rate result. The full 2,402-task evaluation remains pending.

## Target claim

The target is the 79.2% average success rate reported for JEPA-WAM on LIBERO-Plus Table 2 of [arXiv:2608.09381](https://arxiv.org/html/2608.09381). The paper reports results across seven perturbation categories. This project would re-evaluate the released checkpoint. It would not repeat the paper's training.

The [author repository at commit 8bfe3b31a79da57fb217fe10d0e1dc3b282b1c07](https://github.com/SpriteWithoutIce/JEPA_WAM/tree/8bfe3b31a79da57fb217fe10d0e1dc3b282b1c07) gives the command scope as the LIBERO-Plus spatial suite, all categories, one trial per task, and reports the same 79.2% result. The official [LIBERO-Plus repository](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182) says one trial per task is its standard protocol and describes 10,030 perturbed tasks across four suites. Its pinned task classification lists 2,402 spatial-suite variants. The full spatial command is therefore a 2,402-episode run, not a ten-episode check.

The paper says to follow each benchmark's standard evaluation protocol but does not state a trial count in its Appendix B. The author README gives a concrete current command and result mapping. Record this as the operational release-checkpoint target; do not imply that every paper detail is independently specified.

## Training and checkpoint mismatch

The paper reports eight-GPU LIBERO training with global batch size 128 for 60,000 steps. The pinned release README describes an eight-GPU recipe with global batch size 256 for 40,000 steps. Its released policy file is named step-040000-epoch-37-loss=0.0262.pt. These differences mean that a released-checkpoint result can validate the evaluation path and re-check the published score, but cannot reproduce the paper's training run.

The repository and checkpoint are licensed MIT. The required Qwen2.5-0.5B files are Apache-2.0. The official V-JEPA 2 repository states MIT terms for its V-JEPA 2.1 code and models. The LIBERO-Plus dataset card states MIT. These are separate third-party assets; keep their individual terms with the transfer record.

## Pinned release artifacts

The checkpoint repository is [CokeAnd1ce/JEPA_WAM at revision ca10ccbc191d8f56b4346487913e043b2722b6d2](https://huggingface.co/CokeAnd1ce/JEPA_WAM/tree/ca10ccbc191d8f56b4346487913e043b2722b6d2). Its whole repository is reported as 373 GB, so do not clone or download the whole repository.

| Required for released-checkpoint evaluation | Published size | Expected SHA-256 or local verification |
| --- | ---: | --- |
| LIBERO policy checkpoint, step 40,000 | 5,355,388,110 bytes | `e63285fb347048989f14a8a24962a2b921d787f7ada0176a0eacd6b256d57d23` |
| Base VLM checkpoint | 2,525,831,255 bytes | `9bcbe8d0669f020fcc987acb740b09f12da6d90d56d6a95cbf63465f7edead0f` |
| Qwen2.5-0.5B weights | 988,097,824 bytes | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| V-JEPA 2.1 ViT-L/16 384px checkpoint | 5,151,198,524 bytes | No published SHA-256; record the computed local hash |
| LIBERO-Plus simulator assets archive | 6,395,849,578 bytes | `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf` |
| Policy run `config.json` | 3,875 bytes | Local SHA-256 `ed66afc59fb741aad916f2d5424314d756cd7494b8b51acae42c27d4a8e02a19`; pinned Hub Git blob verified |
| Policy run `dataset_statistics.json` | 11,871 bytes | Local SHA-256 `6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3`; pinned Hub Git blob verified |
| Base VLM run `config.json` | 2,075 bytes | Local SHA-256 `293d3bac32bc0bc18db68933a66085a9ee30b48148a17723e79b6b49ad8fe6c8`; pinned Hub Git blob verified |

The policy checkpoint, base VLM, Qwen weights, and LIBERO-Plus archive match their pinned SHA-256 values. The three small JSON metadata files match their pinned Hub Git blob IDs and sizes. The V-JEPA checkpoint has no published SHA-256; its computed local SHA-256 is `7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`, which is a local identity record, not an independent integrity check. The evaluator needs the policy checkpoint and its run metadata, the base VLM and its config, Qwen weights, the V-JEPA checkpoint, and the LIBERO-Plus checkout with its extended assets. It does not need RLDS training data.

At the pinned code revision, V-JEPA checkpoint loading calls torch.load with weights_only=False. Keep any later load inside a disposable, isolated environment and use only the pinned first-party artifact.

## One-task diagnostic

The upstream evaluator completed task 000 in `libero_spatial`, category `Background Textures`, difficulty 2. It used one trial, ten no-op warm-up steps, and one policy action. The task result was 0/1. The capped action does not represent full-task success and must not be compared with the paper's 79.2% result.

The episode took 53.64 seconds. The evaluator's first task iteration took 92.46 seconds, including setup. A straight-line estimate for 2,402 tasks is about 61.7 hours; one task is not enough to predict the full runtime. Do not start the full suite from this estimate. During the run, sampled GPU use was about 6.0–6.6 GiB; peak memory was not measured. The actual EGL-rendered task and policy step completed. The evaluator ran with Python 3.10.12 and PyTorch 2.7.0+cu128 on the RTX 5070. TensorFlow did not register a GPU, but the PyTorch model path used the GPU.

The local LIBERO source calls `torch.load` without a `weights_only` argument for initial states. PyTorch 2.6 and newer default to `weights_only=True`, which rejected the pinned NumPy state file. This run set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the isolated process. PyTorch documents that setting as a compatibility override for call sites that omit the argument; it enables pickle loading and must only be used with trusted, pinned inputs. See [PyTorch serialization environment variables](https://docs.pytorch.org/docs/stable/notes/serialization.html#environment-variables).

The native MagickWand files are under the E:-hosted `native-magick/root/usr` directory. The run used that directory as `MAGICK_HOME` and added its `usr/lib` paths to `LD_LIBRARY_PATH`. It added no system packages. The console record is outside this repository at `E:\codex-research\jepa-wam\evidence\jepa-wam-libero-plus-one-action-diagnostic-attempt-04-20261015.log`; the upstream summary is in its E:-hosted `source/upstream/experiments/logs` directory.

## Host preflight

- The host has an RTX 5070 with 12,227 MiB VRAM.
- The pinned diagnostic environment is under `E:\codex-research\jepa-wam\runtime` (WSL: `/mnt/e/codex-research/jepa-wam/runtime`). It has Python 3.10.12, PyTorch 2.2.0+cu121, and NumPy 1.26.4. PyTorch sees the GPU, but a CUDA matrix operation fails because this wheel has no kernel image for the GPU's `sm_120` architecture.
- A separate diagnostic environment is under `E:\codex-research\jepa-wam\runtime-blackwell`. It has Python 3.10.12 and PyTorch 2.7.0+cu128. The model loaded on the RTX 5070, and the capped evaluator rendered and stepped one LIBERO task.
- The pinned JEPA-WAM and LIBERO-Plus source checkouts, environments, caches, temporary files, model files, extracted assets, and evidence are under E:. All required model files and metadata are complete and verified. The LIBERO-Plus archive passed path checks and extracted successfully. No RLDS training data was downloaded.
- E: has 464 GB free at the latest check. The project checkout remains in its current WSL folder. Its uv environment and cache are on E:. The pre-existing project `.venv` and shared pip cache were left untouched. See [storage and runtime evidence](../evidence/jepa-wam-storage-preflight.md) for locations and disk state.
- The pinned evaluator imports with the E:-hosted Wand overlay, TensorFlow/TFDS dependencies, and pinned `dlimp` revision. TensorFlow does not register a GPU in this environment; PyTorch does. The task renderer and one policy action completed without system package changes.

## Bounded next steps

1. The pinned artifacts, local metadata, source commits, and third-party terms are recorded above. Do not download the 373 GB checkpoint repository or RLDS training data.
2. The upstream one-task diagnostic is complete and produced one action after ten no-op warm-up steps. Its 0/1 task outcome is only pipeline evidence.
3. Keep the 2,402-task spatial suite pending. The first task iteration suggests a long run, and that single-task estimate is not reliable. Agree on a separate run scope and a representative runtime estimate before starting it.
4. If the full run is later approved and completes, preserve all 2,402 task outcomes and report category rates, overall average, runtime, peak GPU memory, checkpoint and source revisions, and deviations. Compare local values with the paper and author README. Keep the training-step and batch-size mismatch explicit.

The bounded diagnostic is complete. The full 2,402-task evaluation remains pending, and no paper success rate has been reproduced locally.
