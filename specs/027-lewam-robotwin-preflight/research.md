# Research: LeWAM RoboTwin 2.0 Preflight

**Reviewed**: 2026-10-15

## Sources and findings

| Source | Finding | Use in this feature |
| --- | --- | --- |
| [LeWAM paper, arXiv:2609.27455](https://arxiv.org/abs/2609.27455), v3 | Reports 92.28% average RoboTwin 2.0 success for LeWAM. | Paper-reported value only; this project has no local score. |
| [Official LeWAM repository](https://github.com/XuejiFang/LeWAM/tree/290458796c2e0725b71cc191b93531a177f97ca7), reviewed `main` revision `290458796c2e0725b71cc191b93531a177f97ca7` | README uses Python 3.10 and `uv`, supports one task or fixed case lists, and requires NVIDIA driver, CUDA 12.x toolkit, C++ compiler, `libvulkan1`, and `ffmpeg`. The default is 50 tasks × two settings × 100 cases, or 10,000 cases by calculation, using seed 42. | Pin the reviewed instructions, identify the bounded first run, and confirm `uv` is compatible with the user's Python-project preference. |
| [GitHub repository metadata](https://api.github.com/repos/XuejiFang/LeWAM) | No repository license metadata is declared. | Treat code and predictor terms as unresolved. Do not state that the encoder license covers them. |
| [Official LeWAM model card](https://huggingface.co/XuejiFang/LeWAM) | The 4.13 GB checkpoint bundle includes a frozen I-JEPA-Huge encoder under CC BY-NC 4.0. The card says this license applies to the encoder and directs users to check terms for the predictor, code, and other components. | Record component-level terms. Check RoboTwin asset terms before any transfer. |
| [RoboTwin 2.0 installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) | Python 3.10 and CUDA 12.1 are listed; WSL is marked unsupported for GPU simulation and rendering. The guide documents Vulkan installation and GPU requirements. | Current host platform is a blocker even if missing packages are installed. |

## Host observations

- `uname -r` reports `6.18.54.1-microsoft-standard-WSL2`.
- `nvidia-smi` reports an NVIDIA GeForce RTX 5070 with 12,227 MiB of memory.
- `nvcc`, `vulkaninfo`, and `libvulkan.so` are absent. `g++` and `ffmpeg` are present.
- `/mnt/e` has 464 GiB free; `/mnt/d` has 578 GiB free.
- No LeWAM source checkout, model checkpoint, or RoboTwin asset was transferred or installed for this preflight.

## Decisions

- Record this as a not-run preflight, not a failed model evaluation.
- Keep the candidate separate from the LeWAM Push-T paper and its pending 450-episode scope.
- Require a supported Linux GPU-rendering host, complete runtime, reviewed component terms, pinned assets, and one declared fixed case list before a first evaluation.
- Report any first fixed-case result only for its named task, setting, and cases. It cannot establish the paper's aggregate 50-task result.
