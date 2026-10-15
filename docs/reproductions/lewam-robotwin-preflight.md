# LeWAM RoboTwin 2.0 Preflight

**Reviewed**: 2026-10-15

**Status**: Research candidate; not run

## Scope

This is a readiness review for evaluating the released LeWAM checkpoint on RoboTwin 2.0. It does not train the model or reproduce a paper result. The paper reports 92.28% average success. This project has no local score.

The official default covers 50 tasks, two settings, and 100 cases per task and setting: 10,000 cases by calculation under seed 42. A future first run must use one named task and a declared fixed case list. Its result would apply only to those cases and cannot reproduce the paper's 50-task average.

This candidate is separate from the pending LeWAM Push-T evaluation in [feature 010](../../specs/010-lewam-pusht-reproduction/spec.md).

## Sources and revisions

| Item | Reviewed record | Status |
| --- | --- | --- |
| Paper | [LeWAM, arXiv:2609.27455](https://arxiv.org/abs/2609.27455), v3 | Paper reports 92.28% average RoboTwin success; no local score. |
| Code | [Official LeWAM repository at `290458796c2e0725b71cc191b93531a177f97ca7`](https://github.com/XuejiFang/LeWAM/tree/290458796c2e0725b71cc191b93531a177f97ca7) | README reviewed. It uses Python 3.10 and `uv`; fixed cases use files named `<task>__<setting>.json`. |
| Model | [Official LeWAM model card](https://huggingface.co/XuejiFang/LeWAM) | Complete bundle is listed as 4.13 GB and includes LeWAM weights, dataset statistics, and frozen I-JEPA-Huge. The encoder is CC BY-NC 4.0. |
| Code terms | [GitHub repository metadata](https://api.github.com/repos/XuejiFang/LeWAM) and repository file list | No project license is declared in repository metadata or the reviewed root file list. The model card says its encoder license does not cover the predictor, code, or other components. Resolve terms before relying on or redistributing these components. |
| Simulator | [RoboTwin 2.0 installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) | The guide lists WSL as unsupported for GPU simulation and rendering. It documents Python 3.10, CUDA 12.1, and Vulkan requirements. |

The paper, code, and model are separate components. Their licenses must not be inferred from the encoder license. RoboTwin simulation asset terms and sizes also need review before transfer.

## Current host

| Requirement | Observation | Gate |
| --- | --- | --- |
| Platform | WSL2 kernel `6.18.54.1-microsoft-standard-WSL2` | **Fail**: RoboTwin documents no WSL GPU simulation or rendering support. |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB | Present; the official guide does not state a minimum VRAM value. |
| CUDA toolkit | `nvcc` absent | **Fail**: LeWAM code requires CUDA 12.x toolkit. |
| Vulkan | `vulkaninfo` and `libvulkan.so` absent | **Fail**: required Vulkan runtime is not available. |
| C++ compiler and FFmpeg | `g++` and `ffmpeg` present | Pass for these two checks only. |
| Storage | E: has 464 GiB free; D: has 578 GiB free | Space is available, but simulator asset size has not been checked. |

## Stop decision

Do not start the evaluator on this host. Installing CUDA or Vulkan inside WSL would not resolve RoboTwin's documented platform limit. No source checkout, model checkpoint, RoboTwin asset, environment, or evaluation output was transferred or created for this preflight.

The code repository has no declared project license. This report does not make a legal determination. Resolve the applicable terms for the code and predictor, and check the RoboTwin assets, before a future evaluation relies on or redistributes them.

## Restart gates

Before a bounded evaluation:

1. Use a host that RoboTwin supports for GPU simulation and rendering, such as a qualified native Linux NVIDIA system.
2. Verify the NVIDIA driver, CUDA 12.x toolkit, C++ compiler, Vulkan runtime, and FFmpeg on that host.
3. Review the terms and sizes for the code, predictor checkpoint, I-JEPA encoder, RoboTwin code, and simulation assets. Record hashes for the selected code and model revisions.
4. Select one task and one setting. Save and hash the official fixed case list; record its case count and seed.
5. Run only that bounded case list first. Preserve raw per-case outcomes and runtime details. Report it as a fixed-case checkpoint evaluation, not as a reproduced 50-task average.

See the [roadmap](../ROADMAP.md#names-that-overlap), [reproduction matrix](../REPRODUCTION_MATRIX.md), and [feature 027](../../specs/027-lewam-robotwin-preflight/spec.md) for project status.
