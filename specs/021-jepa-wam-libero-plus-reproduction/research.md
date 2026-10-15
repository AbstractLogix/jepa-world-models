# Research: JEPA-WAM LIBERO-Plus Checkpoint Re-evaluation

**Reviewed**: 2026-10-15

## Sources and findings

| Source | Finding | Reproduction impact |
| --- | --- | --- |
| [JEPA-WAM paper, arXiv:2608.09381](https://arxiv.org/html/2608.09381) | Table 2 reports 79.2% average success on LIBERO-Plus. The paper describes eight-GPU training with batch 128 for 60,000 steps. Appendix B says to follow the benchmark's standard protocol but does not state the number of trials. | Target a released-checkpoint re-evaluation; do not claim the training run is reproduced. |
| [JEPA-WAM source at commit 8bfe3b31a79da57fb217fe10d0e1dc3b282b1c07](https://github.com/SpriteWithoutIce/JEPA_WAM/tree/8bfe3b31a79da57fb217fe10d0e1dc3b282b1c07) | The MIT repository publishes an evaluation launcher, a one-task, one-action diagnostic, and a README command for the LIBERO-Plus spatial suite, all categories, one trial. Its README reports 79.2% for the seven categories. The release recipe states eight GPUs, batch 256, and 40,000 steps. | The README provides a concrete evaluation command but its released recipe differs from the paper's training details. |
| [JEPA-WAM checkpoint repository at revision ca10ccbc191d8f56b4346487913e043b2722b6d2](https://huggingface.co/CokeAnd1ce/JEPA_WAM/tree/ca10ccbc191d8f56b4346487913e043b2722b6d2) | The card declares MIT. The policy and base VLM files listed by the README are about 5.36 GB and 2.53 GB. The full repository is reported as 373 GB. | Download only the listed evaluation files after the one-step scope and runtime are qualified. |
| [LIBERO-Plus source at commit 4976dc30028e805ff8094b55501d532c48fec182](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182) and [dataset card](https://huggingface.co/datasets/Sylvest/LIBERO-plus) | The benchmark describes 10,030 task variants across four suites and seven perturbation categories. Its evaluation uses one trial per task. The pinned classification file lists 2,402 spatial-suite variants. The dataset card declares MIT. | The author README's spatial command covers 2,402 episodes, not ten episodes. |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | The model card declares Apache-2.0 and lists 988 MB of model weights. | Include its terms and exact file identity in any later transfer record. |
| [V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2) | The repository lists the official V-JEPA 2.1 ViT-L/16 384px checkpoint and states MIT terms for V-JEPA 2.1 code and models. | Use the original required checkpoint and record its hash if transferred. |

## Host and trust boundary

The RTX 5070 has 12 GB VRAM. A Python 3.10.12 environment on E: uses the pinned PyTorch 2.2.0+cu121 and NumPy 1.26.4 versions. PyTorch sees the GPU, but its matrix operation fails because the wheel lacks a kernel image for `sm_120`. A separate E: diagnostic environment uses PyTorch 2.7.0+cu128 and passes that matrix operation. The diagnostic environment differs from the author's tested Python 3.10.16, PyTorch 2.2.0, CUDA 12.1 setup. This check does not confirm model imports, model loading, or benchmark behavior.

The WAM and LIBERO-Plus source checkouts, both virtual environments, pip cache, temporary files, model-cache destination, asset destination, and evidence are under `E:\codex-research\jepa-wam\`. The project checkout remains at `/home/oscar/projects/education/jepa-world-models-tutorial`. The model-cache and asset destinations are empty. The corrected import preflight reaches LIBERO but fails because the `wand` Python package and a discoverable MagickWand shared library are missing. No system packages were added. No checkpoint or simulator asset archive has been downloaded, and LIBERO-Plus rendering remains unverified. See [storage and runtime preflight](../../docs/evidence/jepa-wam-storage-preflight.md) for exact locations and disk state.

At the pinned JEPA-WAM source revision, the V-JEPA loader calls torch.load with weights_only=False. Do not load files until the isolated source checkout and artifact identities are verified.

## Decision

Resolve the missing import dependency and finish the renderer check in the E: diagnostic environment before downloading model or simulator assets. Keep the upstream one-task, one-action diagnostic open until its required artifacts can be loaded safely in that isolated environment. Keep the 2,402-task run pending until the diagnostic passes and its measured cost is known. Do not download training data, other suites, or all files in the Hugging Face repository.
