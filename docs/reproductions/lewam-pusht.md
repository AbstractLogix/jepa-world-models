# LeWAM Push-T Checkpoint Re-evaluation

**Paper results**: reactive policy 85.3% ± 5.0; best-of-32 planning 95.3% ± 2.3; gradient planning 98.7% ± 1.2. These are reported values, not measurements from this project.

**Local status**: Host preflight recorded on 2026-10-14. The runtime readiness gate was not met, so evaluation did not start. Archive transfer and evaluation remain pending.

## Host preflight

| Requirement | Observed | Result |
| --- | --- | --- |
| Host platform | Linux WSL2, kernel 6.18.54.1-microsoft-standard-WSL2, x86_64 | Pass |
| Storage | 748 GiB available on the project and temporary filesystem; the compressed archive is about 13.8 GB and the expanded HDF5 file is about 43.1 GiB | Pass |
| Python | Command-line and existing research-environment Python are 3.13.14. The installed uv interpreters include 3.12, 3.13, and 3.14, but not the Python 3.11 version named by the official installer | Fail |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB, driver 617.42 | Recorded |
| Current evaluation runtime | The existing research environment has PyTorch 2.10.0+cpu; CUDA is unavailable to PyTorch | Not ready |
| MuJoCo and data support | Gymnasium is installed; MuJoCo, h5py, and dm_control are absent from the existing research environment | Fail |
| Headless MuJoCo | libEGL is present. A MuJoCo EGL context was not tested because MuJoCo is absent | Unverified |

The official plan allows a CPU fallback, but this host does not have the pinned research environment or its required MuJoCo and HDF5 packages. No evaluator smoke run or throughput measurement was made. The GPU and driver listing does not establish that the current Python runtime can use CUDA.

## Artifact pins

These are the planned official pins from the source review. They were not downloaded or verified locally.

| Artifact | Official pin | Expected SHA-256 | Local state |
| --- | --- | --- | --- |
| Code | [MinghaoFu/lewam](https://github.com/MinghaoFu/lewam), commit 60732e25ba172b117d0d0a476d3a9dcd5d30c03b | N/A | Not checked out |
| Checkpoint | [LeWAM/lewam-pusht](https://huggingface.co/LeWAM/lewam-pusht), revision d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2; MIT declared by its model card | e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb | Not downloaded or verified |
| Dataset | [LeWAM/lewam-pusht](https://huggingface.co/datasets/LeWAM/lewam-pusht), revision 63346c94f4dbe62b4744261c6c99d63cbff3f6e6; MIT declared by its dataset card | a075dcd49120844a1720cd3f650519f934a632b5c3e9a5a4e413ba8b020aaa43 | Not downloaded or verified |

The dataset archive is 13,804,466,582 bytes and the official guide says it expands to a 43.1 GiB HDF5 file. The official evaluation samples start and goal windows from the training dataset. It is a released-checkpoint re-evaluation, not a training reproduction or held-out generalization test.

## Evaluation status

The planned protocol uses seeds 42, 0, and 1, with 50 episodes per seed and mode, a 25-step goal offset, and a 50-action budget.

| Mode | Paper result | Local episodes | Local rates and comparison |
| --- | --- | ---: | --- |
| Reactive policy | 85.3% ± 5.0 | 0 / 150 | Not measured |
| Best-of-32 planner | 95.3% ± 2.3 | 0 / 150 | Not measured |
| Gradient planner | 98.7% ± 1.2 | 0 / 150 | Not measured |

No local mean can be compared with the paper's reported one-standard-deviation range. The attempt remains blocked before evaluation and pending; no paper result was reproduced locally.

## Resume conditions

Before evaluation, select a run root on E: or D: (for example, WSL `/mnt/e/codex-research/jepa-lewam-pusht`, matching `E:\codex-research\jepa-lewam-pusht`). Set `STABLEWM_HOME` to this root. Place the pinned source checkout, Conda environment and package cache, pip and Hugging Face caches, temporary files, dataset, checkpoint, and evaluator outputs under it. The official evaluator writes results under `$STABLEWM_HOME/eval/`. Then verify the install, MuJoCo headless support, and evaluator startup. No archive, checkpoint, source checkout, or result has been moved or downloaded as part of this path update. Do not start the archive transfer or 450-episode run until the pending scope is reopened.

## Sources

- [LeWAM paper](https://arxiv.org/abs/2610.12818)
- [Pinned official reproduction guide](https://github.com/MinghaoFu/lewam/blob/60732e25ba172b117d0d0a476d3a9dcd5d30c03b/docs/REPRODUCE.md)
- [Official LeWAM project page](https://le-wam.github.io/)
