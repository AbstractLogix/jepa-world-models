# Qantara Push-T Preflight

**Reviewed**: 2026-10-15  
**Status**: Research scoped; no Qantara-specific transfer, conversion, or evaluation run started

## Selected finding

The target is Table 2 of the current Qantara paper, arXiv v2 dated 2026-10-10. It evaluates one Qantara model through three inference paths on Push-T:

| Inference path | Paper success rate |
| --- | ---: |
| Goal-conditioned CEM, K=4 | 90.1 ± 1.1 |
| Goal-blind behaviour cloning, K=2 | 82.1 ± 1.0 |
| Goal-blind video–inverse, K=2 | 83.2 ± 0.6 |

The paper aggregates 50 episodes × five evaluation seeds × three model-training seeds. It reports the population standard deviation across the three training-seed means. Episodes have a 50-step budget. This is a simulated checkpoint evaluation; it does not establish real-robot performance. The values above are paper-reported, not local results.

The paper does not publish the identities of its five evaluation seeds. The official code's timing script lists seeds 42, 2026, and 3407 for a separate timing run, not the five-seed success-rate table. A local fixed five-seed run can repeat the count and evaluator settings, but it cannot match the paper's exact sample list unless the authors provide it.

## Source and artifact gate

| Item | First-party record | Current status |
| --- | --- | --- |
| Paper | [Qantara, arXiv:2607.04978](https://arxiv.org/abs/2607.04978), v2 | Current target reviewed. The paper says its Push-T corpus has 20,000 expert demonstrations and comes from DINO-WM. It describes a roughly 21M-parameter model and simulation-only evaluation. |
| Code | [Official repository at `12429541011e8576eb4fe2bc589d059b0d29dd2e`](https://github.com/corl-team/qantara/tree/12429541011e8576eb4fe2bc589d059b0d29dd2e) | `main` matches this commit. The repository declares MIT. The official configs specify 50 evaluation episodes and a 50-step budget; the same model is called with CEM, BC, and video–inverse dispatches. |
| Checkpoints | [Official checkpoint repository at `de5f7575c9db4a564382e4b2b681fa534dcb3b62`](https://huggingface.co/t-tech/qantara-checkpoints/tree/de5f7575c9db4a564382e4b2b681fa534dcb3b62) | The card declares MIT and releases Qantara Push-T checkpoints for seeds 11, 22, and 33. The three files total about 275 MB. Hub metadata advertises hashes in [feature 019](../../specs/019-qantara-multiparadigm/research.md); no file was downloaded or locally hash-verified. Each file is a full serialized Python model object. Load only after reviewing the pinned source and in a controlled environment. |
| Paper corpus | [DINO-WM dataset at `d5c69e8fe60c3ed6b01d2f6e88fbfd66be078e57`](https://huggingface.co/datasets/gaoyuezhou/dino_wm/tree/d5c69e8fe60c3ed6b01d2f6e88fbfd66be078e57) | The card declares MIT. The `pusht_noise.zip` file is 2,785,304,515 bytes; its Hub SHA-256 is `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08`. A matching archive was already staged at `/tmp/lpwm-data/pusht_noise.zip` before this Qantara preflight; its local SHA-256 matches. This task did not transfer or unpack it. Its ZIP listing has 18,685 training and 21 validation MP4 path entries; these are filename counts, not verified Qantara episode identities. |
| HDF5 input | [LeWM Push-T dataset at `655cd446b9929369d7d406001da85c15d1457850`](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850) | The card declares MIT. `pusht_expert_train.h5.zst` is 13,136,247,974 bytes, with Hub SHA-256 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`. Qantara's official config requests the HDF5 dataset name `pusht_expert_train`, but its README says LeRobot / D4RL while the paper names the DINO-WM corpus. Exact episode identity and conversion are unresolved. Do not assume the files are interchangeable. This task did not transfer the LeWM HDF5 file. |
| Runtime | [Official Qantara installation notes](https://github.com/corl-team/qantara/blob/12429541011e8576eb4fe2bc589d059b0d29dd2e/README.md) and pinned `pyproject.toml` | Qantara declares Python 3.10–3.13, PyTorch 2.7.x, and MuJoCo 3.2.0. Its README lists EGL as the default and OSMesa as a fallback. The local Python 3.11 H-JEPA environment has CUDA 12.8 and PyTorch 2.7.1, but MuJoCo is 3.14. EGL initialization exits 139. A direct `MUJOCO_GL=osmesa` import fails in PyOpenGL before an environment starts. Qantara's evaluator runtime is not qualified. |

### Code-level data identity check

At the pinned Qantara code revision, the Push-T training config requests the HDF5 dataset `pusht_expert_train` with `pixels`, `action`, `proprio`, and `state` fields. The evaluator loads that dataset through `stable_worldmodel.data.HDF5Dataset`, selects rows by `episode_idx`/`ep_idx` and `step_idx`, then samples start and goal states from those rows. The pinned Qantara repository contains no dataset-conversion script. The [official DINO-WM README at `0a9492fa12044b852ae9e001cc74604b79c8bb0c`](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/README.md) documents its release under a `pusht_noise` directory, but does not define a conversion to Qantara's HDF5 schema or identify matching episode IDs. This confirms a format and provenance gap; it does not prove that the underlying trajectories differ.

### Adjacent conversion path

The pinned [H-JEPA converter](https://github.com/kevinghst/H-JEPA/blob/97054ecaff83e13839195ec4b92d74c08a0127c5/scripts/data/convert_pusht_noise_to_h5.py) reads DINO-WM `states.pth`, `velocities.pth`, `rel_actions.pth`, `seq_lengths.pkl`, and numbered observation videos. It writes HDF5 fields that include Qantara's required `pixels`, `action`, `proprio`, and `state`, plus episode and step indexes. It assigns episode IDs from sequence order, steps from `0..L-1`, and scales relative actions by `0.01`. This is a schema-compatible candidate path, not proof that Qantara used it. The converter's `torch.load(..., weights_only=False)` and `pickle.load()` calls also require source and artifact review before execution.

At the pinned Qantara revision, its README says Push-T data comes from LeRobot / D4RL expert data, while the paper names the DINO-WM corpus. H-JEPA's README describes its HDF5 input as downloaded from LeWM. No first-party Qantara record links its actual training/evaluation HDF5 file to this H-JEPA converter, establishes source episode IDs, or confirms transformation and row order. The local raw archive's hash is verified, but that does not resolve this mapping. T003 remains open. No Qantara asset was transferred or converted for this evaluation, and no checkpoint was transferred.

The project has an RTX 5070 with 12 GB of VRAM. CUDA visibility does not clear the renderer gate. No Qantara evaluation command was run. The H-JEPA OSMesa import result is a cross-project renderer check, not a Qantara episode result.

## Stop conditions

Before transfer or evaluation:

1. Resolve the lineage of the actual Qantara HDF5 training/evaluation file. Compare source IDs, episode counts, sequence lengths, transformations, row order, and hashes. Treat the H-JEPA conversion script only as a candidate until Qantara's use is established. Do not substitute an unverified dataset.
2. Recheck license terms, revisions, sizes, and every checkpoint/data hash. Confirm expanded size and available storage before transfer; preserve assets outside Git.
3. Review the pinned source and serialized checkpoint contents before loading the full Python objects. Define a controlled loading procedure.
4. Qualify the exact Python, PyTorch, stable-worldmodel, MuJoCo, CUDA, and headless renderer combination with one case. Record software versions, hardware, time, and peak memory.
5. Declare five evaluation seeds before the run. Since the paper does not name its seed identities, label any different seed set a protocol replication and disclose the sampling difference.
6. If all gates pass, run all three checkpoint seeds and three inference paths on the same 50-episode cases for each of five evaluation seeds. Keep every raw episode and failure outside Git.

No paper result is reproduced by this preflight. See [feature 019](../../specs/019-qantara-multiparadigm/spec.md), the [roadmap](../ROADMAP.md), and the [reproduction matrix](../REPRODUCTION_MATRIX.md).
