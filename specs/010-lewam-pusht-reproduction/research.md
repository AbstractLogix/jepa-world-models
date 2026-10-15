# Research: Released LeWAM Push-T Checkpoint Reproduction

**Reviewed**: 2026-10-14

## Decisions

### Select the end-to-end LeWAM paper and one named result

The selected paper is *LeWAM: End-to-End World and Action Modeling with JEPAs*, arXiv:2610.12818. Its official project page reports a Push-T result for a released checkpoint. The official reproduction guide names three evaluation seeds (42, 0, 1), 50 episodes per seed, a 25-step goal offset, and a 50-action budget. It lists reactive policy success of 85.3% ± 5.0, best-of-32 success of 95.3% ± 2.3, and gradient planner success of 98.7% ± 1.2.

This paper is distinct from the other October 2026 preprint with a similar title, *LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC* (arXiv:2610.12407). This feature does not claim to evaluate that paper.

### Use the official source and released artifacts

- Code repository: [`MinghaoFu/lewam`](https://github.com/MinghaoFu/lewam), pinned at `60732e25ba172b117d0d0a476d3a9dcd5d30c03b`.
- Reproduction guide: [`docs/REPRODUCE.md`](https://github.com/MinghaoFu/lewam/blob/60732e25ba172b117d0d0a476d3a9dcd5d30c03b/docs/REPRODUCE.md).
- Push-T checkpoint: [`LeWAM/lewam-pusht`](https://huggingface.co/LeWAM/lewam-pusht), revision `d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2`. The model card declares MIT; `lewam_best.pt` is 63,906,043 bytes with SHA-256 `e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb`.
- Push-T dataset: [`LeWAM/lewam-pusht`](https://huggingface.co/datasets/LeWAM/lewam-pusht), revision `63346c94f4dbe62b4744261c6c99d63cbff3f6e6`. The dataset card declares MIT; `pusht.h5.zst` is 13,804,466,582 bytes with SHA-256 `a075dcd49120844a1720cd3f650519f934a632b5c3e9a5a4e413ba8b020aaa43`.
- The model card says the weights came from `lewam-jointflow@2d52888`, were converted to the v1 schema, and retain exactly matching tensors apart from an obsolete all-zero `null_goal` tensor. It reports that strict loading and the key model paths passed.
- The official data guide says the archive expands to a 43.1 GiB HDF5 file. It describes a larger uncompressed-frame cache for training. Evaluation should use the official configured data path without creating that cache unless required by the evaluator.

### Re-evaluate the released checkpoint, do not retrain

The local result measures whether the released checkpoint and official evaluator reproduce a published evaluation. It does not reproduce pretraining or LeWAM training. The official guide states that goal-reaching evaluation samples start/goal windows from the dataset used for training. Do not describe the sampled episodes as held-out generalization.

### Report all runs and keep source claims separate

Preserve all episode outcomes and report success by seed and mode, then summarize across the three seeds. Link the paper's values next to the local values. Label the local mean as within or outside the paper's reported mean ± one standard deviation. This is a comparison rule, not a statistical test. Report failures and deviations. Do not infer statistical equivalence or general validity from a small evaluator repeat.

## Alternatives Considered

- **Train LeWAM from scratch**: rejected for this feature. It would add a large training and data-compute requirement and is not needed to check the published checkpoint evaluation.
- **Run only a few episodes and call it a reproduction**: rejected. A smaller run can be a smoke check but remains partial and cannot validate the paper's three-seed summary.
- **Use point-cloud JEPA as the first exact attempt**: deferred. Its official evaluator states a CUDA 12 and 24 GB GPU requirement; this host has a 12 GB GPU and its current PyTorch runtime is CPU-only.
- **Use the synthetic CartPole notebook as validation**: rejected. It does not test the published Push-T claim.
