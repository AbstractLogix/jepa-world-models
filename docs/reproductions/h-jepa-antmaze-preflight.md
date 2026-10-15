# H-JEPA Visual AntMaze Preflight

**Reviewed**: 2026-10-15
**Status**: No task completed; MuJoCo renderer startup remains blocked on this host

## Selected result

The v1 paper reports 73.3% ± 3.5 for three-level H-JEPA on Visual AntMaze and 18.0% ± 3.5 for flat LeWM. Table 1 reports the three-level hierarchical planner as a reference result; the paper reports mean ± standard error over three model seeds. The authors' evaluation uses 50 fixed start/goal tasks per environment. This is a simulated planning result, not a physical-robot result.

The local target is a released-checkpoint evaluation of the H-JEPA and LeWM AntMaze models for seeds 42, 43, and 44. Use the official task file and planner settings. Keep every task outcome, per-seed score, runtime, and memory measurement. This evaluates published checkpoints; it does not independently reproduce model training.

## Source and artifact gate

| Item | First-party record | Current status |
| --- | --- | --- |
| Paper | [H-JEPA, arXiv:2610.06805 v1](https://arxiv.org/abs/2610.06805v1) | Submitted 2026-10-05. The paper reports the selected AntMaze result and three-seed protocol. |
| Code | [Official H-JEPA repository](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5) and its [pinned project configuration](https://github.com/kevinghst/H-JEPA/blob/97054ecaff83e13839195ec4b92d74c08a0127c5/pyproject.toml) | Revision `97054ecaff83e13839195ec4b92d74c08a0127c5`; the repository `LICENSE` is MIT. The project pins Python 3.11, PyTorch 2.7.1, MuJoCo 3.14.0, and OGBench 1.2.1. Training and evaluation scripts are present. |
| Checkpoints | [Official checkpoint repository](https://huggingface.co/jepa-world-models/h-jepa/tree/6844abd35b5e5e5a29b5c9d506c62330bc0742f5) | Revision `6844abd35b5e5e5a29b5c9d506c62330bc0742f5`; its card declares MIT. Seed-42 LeWM (`49,592,431` bytes) has SHA-256 `488400edfb13ac718fb400115a756d057472b06ccf7185d653271d844cee5ca9`; H-JEPA-L3 (`139,538,683` bytes) has SHA-256 `c55d9835d6c774da4363c45f8c390c8abed5a496282dd5b9815615e417f69f99`. Both normalizers have SHA-256 `1e86ce0a72836c69455a73f614765b7c186bebeca0904be60a9db12ed889fb44`. Seeds 43 and 44 were not transferred. |
| Evaluation tasks | [Official dataset card and files](https://huggingface.co/datasets/jepa-world-models/h-jepa/tree/cc05cb8d2aaf9aec0d5364d073c6369ff760b60b) | Dataset revision `cc05cb8d2aaf9aec0d5364d073c6369ff760b60b`; its card declares CC BY 4.0. The fixed AntMaze task file has 50 start/goal tasks, is 155,119,789 bytes, and has SHA-256 `72325de082844e0fb85459ae970c2197dd9030edb1bab3891be9a18d465f4362`. It is staged outside the repository. Evaluation does not need the AntMaze training corpus. |
| Runtime and terms | The official README specifies Python 3.11.13 and CUDA 12.8 wheels. [OGBench](https://github.com/seohongpark/ogbench) is MIT licensed and credits D4RL for the AntMaze environment; [D4RL](https://github.com/Farama-Foundation/D4RL) licenses code under Apache 2.0 and, unless noted otherwise, datasets under CC BY 4.0. The [MuJoCo Python package](https://github.com/google-deepmind/mujoco/blob/main/python/pyproject.toml) declares Apache 2.0. | An isolated Python 3.11.13 environment is installed. PyTorch 2.7.1+cu128 reports CUDA 12.8 and sees the RTX 5070 (12,227 MiB); MuJoCo is 3.14.0 and OGBench is 1.2.1. The EGL renderer crashes during initialization. The installed OGBench Ant XML has no separate license header; retain OGBench and D4RL attribution and do not copy it into this project. No simulation task completed, so evaluation memory and runtime are not measured. |

The checkpoint and source licenses do not replace review of third-party code, simulator, and asset terms. Do not use the unrelated DROID dataset for this AntMaze result.

## Local smoke attempt (2026-10-15)

The official source at `97054ecaff83e13839195ec4b92d74c08a0127c5`, the seed-42 checkpoints, normalizers, and task file are in `/tmp/codex-hjepa-antmaze-97054`; their hashes match the pinned records above. No training corpus was downloaded. The isolated environment uses Python 3.11.13, PyTorch 2.7.1+cu128, CUDA 12.8, MuJoCo 3.14.0, and OGBench 1.2.1. The checkpoint's `StandardScaler` records scikit-learn 1.7.2. The first load under 1.9.1 raised a version warning, so the temporary environment now pins 1.7.2. The evaluator was not rerun after this pin because EGL fails before simulation starts.

The official LeWM seed-42 command used `--config-name ant_flat`, `eval.num_eval=1`, and the pinned checkpoint. It loaded the checkpoint, logged `MUJOCO_GL=egl, attempting to import specified OpenGL backend`, then exited with code 139 before an episode result. A direct EGL context check also exited with code 139. With only the NVIDIA EGL vendor selected, the host reports zero EGL devices. The H-JEPA one-task command was not started because both models use the same renderer path. No success value, episode, memory measure, or benchmark runtime was produced. A second smoke with `MUJOCO_GL=osmesa` failed during MuJoCo import: PyOpenGL had no active GL module and raised `AttributeError` before an environment could start. The host has no working EGL or OSMesa path in this runtime.

### Follow-up renderer diagnostics (2026-10-15)

The isolated Python 3.11.13 runtime and staged seed-42 assets remain available. A tiny MuJoCo scene also failed before producing a frame in these renderer-only checks:

- EGL with the Mesa vendor and a surfaceless platform: exit 139.
- EGL with the Mesa vendor on WSLg Wayland using the D3D12 Gallium driver: exit 139.
- GLFW on the WSLg X display: exit 139. This was a diagnostic only; the official evaluator still selects EGL.

The base Python 3.13 environment has none of the MuJoCo evaluation packages. The host exposes `/dev/dxg`, but the `/dev/dri` nodes are root-owned with mode `0600`. These observations do not identify the crash cause. No package, official source, checkpoint, task file, or evaluation setting was changed. No benchmark task started, and the renderer gate remains open.

Command, run from `/tmp/codex-hjepa-antmaze-97054/h_jepa`:

```sh
HJEPA_HOME=/tmp/codex-hjepa-antmaze-97054/data /tmp/codex-hjepa-antmaze-97054/.venv/bin/python eval.py --config-name ant_flat seed=42 eval.num_eval=1 output.dir=/tmp/codex-hjepa-antmaze-97054/results/lewm-flat-seed42-smoke policy=/tmp/codex-hjepa-antmaze-97054/data/ckpts/ant/ant_lewm/seed42/ant_lewm_object.ckpt
```

The task file and the two seed-42 model checkpoints remain in `/tmp`, outside Git. The official Push-T archive was not transferred.

## Stop conditions

Before a full evaluation:

1. Resolve the host EGL failure without changing the official renderer or task semantics. Repeat one task for each seed-42 model and record memory and time.
2. If both smoke runs pass, transfer and verify the four seed-43/44 model checkpoints and normalizers.
3. Evaluate all 50 tasks for seeds 42, 43, and 44 with the paper's planner settings. Save raw task outcomes outside Git and report per-seed success and mean ± standard error.
4. Label any completed result as released-checkpoint evaluation. Do not claim independent training or general physical-robot performance.

No task outcome is available. See the [roadmap](../ROADMAP.md), [reproduction matrix](../REPRODUCTION_MATRIX.md), and [project Spec Kit tasks](../../specs/001-project-vision/tasks.md) for its future status.
