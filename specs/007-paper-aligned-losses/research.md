# Research: Paper-aligned JEPA objectives

**Reviewed**: 2026-10-14

## Primary research

### LeJEPA

- Paper: Balestriero and LeCun, [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/html/2511.08544), arXiv version 3, dated 2025-11-14.
- First-party code: [galilai-group/lejepa](https://github.com/galilai-group/lejepa), reviewed `HEAD` `c293d291ca87cd4fddee9d3fffe4e914c7272052` on 2026-10-14.
- The paper defines SIGReg as an average over random unit directions of the Epps-Pulley goodness-of-fit statistic. For each direction, it compares the empirical characteristic function of projected embeddings with the characteristic function of a standard normal, weights squared error by the Gaussian characteristic function, integrates over `[-5, 5]`, and multiplies by batch size.
- The paper's starting settings are `lambda=0.05`, two global and eight local views, batch size at least 128, 1,024 directions, and 17 integration points over `[-5, 5]`. Its objective combines view invariance and SIGReg. It reports results across more than ten datasets and sixty architectures; its abstract gives 79% frozen linear evaluation on ImageNet-1K with ViT-H/14.
- Current notebook 02 has an Epps-Pulley sketch with 128 projections, no explicit per-view SIGReg aggregation, and no recorded training-seed comparison. This is close to the paper's loss family but is a reduced implementation.

### Rectified LpJEPA

- Paper: Kuang et al., [Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations](https://arxiv.org/html/2602.01456), arXiv version 2, dated 2026-05-28.
- First-party code: [YilunKuang/rectified-lp-jepa](https://github.com/YilunKuang/rectified-lp-jepa), reviewed `HEAD` `5ae61ab49a4ae489c68f2328ba614472163518d6` on 2026-10-14. The repository includes a standalone CIFAR-100/ResNet-18 example and a full ImageNet-100 training path.
- RDMReg matches projected feature samples to projected samples from a Rectified Generalized Gaussian target. It uses normalized random directions, sorts each projected batch, and averages the squared paired differences. The default `p=1`, `mu=0` target is Rectified Laplace with scale `1/sqrt(2)`. Its projector ends with ReLU.
- The paper's main ImageNet-100 setup uses a ResNet-50 with a three-layer 2,048-wide projector, 224-pixel inputs, 1,000 epochs, LARS, batch size 128, 8,192 directions, and weights 25 for invariance and 125 for RDMReg. It reports one training run (seed 5) on one NVIDIA L40S GPU, with about 2 days and 7 hours of pretraining.
- In Table 1, the paper reports ImageNet-100 encoder top-1 accuracy of 84.72% and L0 sparsity of 0.6940 for its Rectified Laplace setting; its LeJEPA baseline reports 84.80% and L0 sparsity of 1.0000. These values are paper results, not local targets or measurements.
- The paper defines L0 sparsity as the fraction of nonzero features. Its L1 sparsity is `(1/D) * (||z||_1 / ||z||_2)^2` per sample. Lower values mean sparser features. The local report must pair sparsity with linear-probe accuracy because all-zero features also have a low L0 value.
- Current notebook 03 uses the project's variance/covariance helper and an L1/entropy proxy. It does not sample a Rectified Generalized Gaussian target, perform sliced Wasserstein matching, or explicitly rectify the projector output. The saved prior output reports chance-level probe accuracy and near-zero variance; the matched run will be rerun and that prior output will not count as evidence for the new protocol.

## Local study decisions

- Keep the published ImageNet-100 benchmarks out of the local claim. The local study uses CIFAR-10, a roughly 455,000-parameter four-stage CNN, 32-pixel inputs, 640 unlabeled training images (reduced from 1,000 after the first CPU run exceeded its 900-second cell limit), 20 epochs, AdamW, and five seeds. It is an objective study, not a scaled reproduction of the ImageNet experiments.
- For SIGReg, keep the current notebook's 128-projection and `lambda=0.1` teaching arm. The paper-aligned arm uses the exact Epps-Pulley expression, per-view regularization, 512 projections, 17 points over `[-5, 5]`, and `lambda=0.05`. The paper recommends 1,024 directions; 512 is a recorded local reduction.
- For RDMReg, keep the current helper losses as the teaching arm. The paper-aligned arm uses the Rectified Laplace target and exact sliced 2-Wasserstein matching with 512 projections. The lower projection count and small local architecture are explicit deviations from the ImageNet-100 setup.
- Use local deterministic generators for projections and target samples. A run seed controls model initialization, data order, augmentations, and loss sampling.
- Use the same fixed CIFAR-10 indices across paired methods. The project uses the official CIFAR-10 source. Its reviewed source page asks users to cite the dataset report but does not state license terms; the repository will not redistribute the data.

## Alternatives considered

- **Install and run each full upstream training stack**: rejected for this feature because the paper-scale runs need ImageNet-100, much larger training budgets, GPU resources, and extra dependencies. That work remains a separate reproduction task.
- **Rename the existing proxy helpers as SIGReg/RDMReg**: rejected because it would make their behavior unclear and could silently change other lessons. Keep proxy names and add separate paper-aligned functions.
- **Use a single seed**: rejected because the project evaluation protocol requires at least five training seeds for a comparison unless the result is clearly exploratory.

## Source list

- [LeJEPA paper, arXiv v3](https://arxiv.org/html/2511.08544)
- [LeJEPA official repository](https://github.com/galilai-group/lejepa)
- [Rectified LpJEPA paper, arXiv v2](https://arxiv.org/html/2602.01456)
- [Rectified LpJEPA official repository](https://github.com/YilunKuang/rectified-lp-jepa)
- [CIFAR-10 official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Project evaluation protocol](../../docs/EVALUATION_PROTOCOL.md)
