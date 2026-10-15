# Point-Cloud JEPA Push-T Reproduction

**Paper result**: Point-LeWM 83.6% ± 3.4; Point-Delta-JEPA 70.8% ± 7.7. These are the paper's reported values, not measurements from this project.  
**Local status**: Stopped at host preflight on 2026-10-14. No evaluation assets were downloaded.

## Host preflight

| Requirement | Observed | Result |
| --- | --- | --- |
| Linux x86-64 | WSL2 Linux, kernel `6.18.54.1-microsoft-standard-WSL2`, x86-64 | Pass |
| CUDA 12 evaluation runtime | PyTorch `2.10.0+cpu`; `torch.version.cuda` is `None`; `torch.cuda.is_available()` is `False` | Fail |
| EGL | `libEGL.so.1` is present and `DISPLAY=:0`; an EGL device context was not tested | Unverified |
| GPU memory | NVIDIA GeForce RTX 5070, 12,227 MiB | Fail; below the official 24 GB requirement |
| NVIDIA driver | `617.42` | Recorded; this does not provide a CUDA-enabled PyTorch runtime by itself |

The official evaluation requires Linux x86-64, CUDA 12, EGL, and a GPU with at least 24 GB of memory. This host fails the CUDA-runtime and memory gates. The run stopped before creating an upstream checkout, installing its environment, or downloading the 4,616,741,240-byte v1 dataset archive and checkpoints. No paper result was reproduced locally.

## Pinned protocol

The selected experiment compares the released Point-LeWM and Point-Delta-JEPA checkpoints on the same Push-T windows. The planned protocol uses the official code, v1 dataset, and checkpoint revisions in [feature 011](../../specs/011-pointcloud-jepa-reproduction/spec.md): ten seeds (`1, 25, 35, 36, 40, 42, 65, 80, 83, 99`) and 50 episodes per seed for each method. The report remains incomplete until a supported host completes all 1,000 episode outcomes.

## Sources

- [Paper, Table 2](https://arxiv.org/html/2608.29434v2)
- [Official reproduction guide](https://github.com/fafraob/point-lewm/blob/main/docs/reproduction.md)
- [Official code repository](https://github.com/fafraob/point-lewm)
