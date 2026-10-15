# LeJEPA and Rectified LpJEPA local objective comparison

**Reviewed**: 2026-10-14

## Claim status

This report records two locally measured objective comparisons. They test the local loss implementations on small CIFAR-10 runs. They do **not** reproduce the ImageNet-1K LeJEPA result or the ImageNet-100 Rectified LpJEPA result, and they do not validate every claim in either paper.

## Paper-reported results and sources

| Paper | Paper-reported result | Scope |
| --- | --- | --- |
| [LeJEPA, arXiv v3](https://arxiv.org/html/2511.08544) | 79% frozen linear evaluation on ImageNet-1K | The paper uses a ViT-H/14 for this result. Its first-party [code](https://github.com/galilai-group/lejepa/commit/c293d291ca87cd4fddee9d3fffe4e914c7272052) was reviewed at commit `c293d291ca87cd4fddee9d3fffe4e914c7272052`. |
| [Rectified LpJEPA, arXiv v2](https://arxiv.org/html/2602.01456) | Rectified Laplace setting: 84.72% encoder top-1 and 0.694 L0 fraction nonzero; the LeJEPA baseline reports 84.80% and 1.000 L0 fraction nonzero. | ImageNet-100, one reported seed (5), ResNet-50, 2,048-wide three-layer projector, 224-pixel images, 1,000 epochs, LARS, batch 128, and 8,192 projections. The first-party [code](https://github.com/YilunKuang/rectified-lp-jepa/commit/5ae61ab49a4ae489c68f2328ba614472163518d6) was reviewed at commit `5ae61ab49a4ae489c68f2328ba614472163518d6`. |

The paper values above are source-reported, not local measurements. The local input data is the [official CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset page requests citation of its technical report and does not state license terms; this repository does not redistribute the dataset.

## Local protocol

- Dataset: CIFAR-10. Data selection seed `20261014`. Use 640 balanced self-supervised train images (64 per class), 1,000 disjoint balanced train images for the frozen probe, and 1,000 balanced held-out test images.
- The first 1,000-image self-supervised run exceeded the 900-second notebook-cell limit before it saved its results. The final study uses 640 self-supervised images. Both notebooks use this same fixed split size and report that local limit.
- Training: 20 epochs, batch size 128, AdamW learning rate `0.003`, weight decay `1e-05`, and seeds `[7, 17, 27, 37, 47]`.
- Probe: frozen features; scikit-learn logistic regression with `max_iter=1000`; 1,000 labeled train images and 1,000 held-out test images. The probe limit was raised after the first clean run showed convergence warnings. The final clean runs had no convergence warnings.
- Aggregates: mean, median, and two-sided 95% Student-t intervals over five training seeds. Paired intervals compare the same seed across the teaching and paper-aligned arms. All intervals are uncertain at this run count.
- Runtime environment: Python 3.13.14, PyTorch 2.10.0+cpu, TorchVision 0.25.0+cpu, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1; Linux WSL2, CPU, 8 PyTorch intra-op threads. Notebook model sizes: 454,688 parameters for Notebook 02 and 653,088 total parameters for Notebook 03.
- Inputs are 32×32 RGB images, normalized with CIFAR-10 channel means `(0.4914, 0.4822, 0.4465)` and standard deviations `(0.2470, 0.2430, 0.2610)`. Notebook 02 uses global crop scale `[0.6, 1.0]` and local crop scale `[0.2, 0.6]`; Notebook 03 uses two crops with scale `[0.2, 1.0]`. Both use random horizontal flips. The local working tree was not committed for these runs. CPU model and peak memory were not recorded.

### Objective settings

- Notebook 02 teaching arm: two global plus four local views, 128 random projections, `lambda=0.1`, current sketch aggregation. Paper-aligned arm: same view set, Epps–Pulley SIGReg per view, 512 random projections, 17 quadrature points over `[-5, 5]`, and `lambda=0.05`. LeJEPA recommends two global plus eight local views and 1,024 projections; this local run uses fewer.
- Notebook 03 uses the same 256-dimensional CNN backbone in both arms and a 256-wide, three-layer projector with a final ReLU in both arms. The teaching arm uses the existing variance/covariance and L1/entropy proxies. The paper-aligned arm uses Rectified Laplace RDMReg with `p=1`, `mu=0`, scale `1/sqrt(2)`, 512 projections, and weights 25 (invariance) and 125 (RDMReg).
- Local differences from the Rectified LpJEPA ImageNet-100 recipe include dataset, data scale, network, projector width, input size, optimizer, epoch count, projection count, and number of seeds.

## Notebook 02: LeJEPA

| Arm | Seed | Probe top-1 | Mean feature std | Covariance trace | Mean abs. off-diagonal covariance | Run seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teaching sketch | 7 | 26.00% | 1.4599 | 628.57 | 1.05422 | 107.8 |
| Teaching sketch | 17 | 26.90% | 1.6699 | 804.81 | 1.33641 | 112.7 |
| Teaching sketch | 27 | 26.60% | 1.3876 | 549.31 | 0.79434 | 89.7 |
| Teaching sketch | 37 | 28.20% | 1.3008 | 479.78 | 0.68193 | 86.1 |
| Teaching sketch | 47 | 29.90% | 1.3921 | 574.19 | 1.01152 | 80.8 |
| Paper-aligned SIGReg | 7 | 26.80% | 0.8594 | 203.72 | 0.24358 | 91.0 |
| Paper-aligned SIGReg | 17 | 27.10% | 0.9358 | 238.25 | 0.26986 | 90.9 |
| Paper-aligned SIGReg | 27 | 26.90% | 0.8921 | 222.33 | 0.26189 | 92.2 |
| Paper-aligned SIGReg | 37 | 27.70% | 0.9310 | 238.19 | 0.28333 | 93.9 |
| Paper-aligned SIGReg | 47 | 27.90% | 0.9043 | 225.24 | 0.26537 | 98.0 |

### Notebook 02 aggregates

| Arm | Metric | Mean | Median | 95% t interval |
| --- | --- | ---: | ---: | ---: |
| Teaching sketch | Probe top-1 | 27.52% | 26.90% | [25.59%, 29.45%] |
| Teaching sketch | Mean feature std | 1.442 | 1.392 | [1.269, 1.615] |
| Teaching sketch | Covariance trace | 607.333 | 574.195 | [455.030, 759.635] |
| Teaching sketch | Mean absolute off-diagonal covariance | 0.976 | 1.012 | [0.661, 1.290] |
| Paper-aligned SIGReg | Probe top-1 | 27.28% | 27.10% | [26.67%, 27.89%] |
| Paper-aligned SIGReg | Mean feature std | 0.905 | 0.904 | [0.866, 0.943] |
| Paper-aligned SIGReg | Covariance trace | 225.544 | 225.238 | [207.895, 243.194] |
| Paper-aligned SIGReg | Mean absolute off-diagonal covariance | 0.265 | 0.265 | [0.247, 0.283] |

### Paired differences (paper-aligned minus teaching)

| Metric | Mean difference | 95% paired t interval |
| --- | ---: | ---: |
| Probe top-1 | -0.240 percentage points | [-1.591, 1.111] percentage points |
| Mean feature std | -0.53755 | [-0.70758, -0.36752] |
| Covariance trace | -381.78816 | [-533.51895, -230.05738] |
| Mean absolute off-diagonal covariance | -0.71088 | [-1.03171, -0.39005] |

## Notebook 03: Rectified LpJEPA

L0 is the fraction of nonzero projector features. L1 is `(1/D) * (||z||1 / ||z||2)^2` per sample; lower values indicate more sparse features. A zero vector has L1 value zero, so interpret both measures beside probe accuracy and the zero-vector fraction.

| Arm | Seed | Encoder probe | Projector probe | L1 sparsity | L0 fraction nonzero | Zero-vector fraction | Run seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teaching proxy | 7 | 31.10% | 10.00% | 0.00000 | 0.00000 | 1.000 | 20.5 |
| Teaching proxy | 17 | 33.50% | 17.10% | 0.00624 | 0.00636 | 0.132 | 20.6 |
| Teaching proxy | 27 | 31.10% | 16.60% | 0.00986 | 0.01204 | 0.000 | 19.2 |
| Teaching proxy | 37 | 33.30% | 19.30% | 0.01600 | 0.01899 | 0.000 | 21.5 |
| Teaching proxy | 47 | 33.10% | 14.50% | 0.01325 | 0.01521 | 0.000 | 21.9 |
| Paper-aligned RDMReg | 7 | 32.90% | 22.50% | 0.27509 | 0.41971 | 0.000 | 21.7 |
| Paper-aligned RDMReg | 17 | 33.40% | 22.70% | 0.28053 | 0.43506 | 0.000 | 33.5 |
| Paper-aligned RDMReg | 27 | 31.80% | 22.50% | 0.26858 | 0.41310 | 0.000 | 22.2 |
| Paper-aligned RDMReg | 37 | 31.60% | 23.40% | 0.27462 | 0.42089 | 0.000 | 22.5 |
| Paper-aligned RDMReg | 47 | 32.70% | 21.10% | 0.28168 | 0.43532 | 0.000 | 22.0 |

### Notebook 03 aggregates

| Arm | Metric | Mean | Median | 95% t interval |
| --- | --- | ---: | ---: | ---: |
| Teaching proxy | Encoder probe top-1 | 32.42% | 33.10% | [30.91%, 33.93%] |
| Teaching proxy | Projector probe top-1 | 15.50% | 16.60% | [11.13%, 19.87%] |
| Teaching proxy | Projector L1 sparsity | 0.009 | 0.010 | [0.001, 0.017] |
| Teaching proxy | Projector L0 fraction nonzero | 0.011 | 0.012 | [0.001, 0.020] |
| Paper-aligned RDMReg | Encoder probe top-1 | 32.48% | 32.70% | [31.54%, 33.42%] |
| Paper-aligned RDMReg | Projector probe top-1 | 22.44% | 22.50% | [21.40%, 23.48%] |
| Paper-aligned RDMReg | Projector L1 sparsity | 0.276 | 0.275 | [0.270, 0.283] |
| Paper-aligned RDMReg | Projector L0 fraction nonzero | 0.425 | 0.421 | [0.412, 0.437] |

### Paired differences (paper-aligned minus teaching)

| Metric | Mean difference | 95% paired t interval |
| --- | ---: | ---: |
| Encoder probe top-1 | 0.060 percentage points | [-1.556, 1.676] percentage points |
| Projector probe top-1 | 6.940 percentage points | [2.918, 10.962] percentage points |
| Projector L1 sparsity | 0.26703 | [0.25703, 0.27703] |
| Projector L0 fraction nonzero | 0.41429 | [0.39910, 0.42949] |

## Interpretation and limits

- Notebook 02 probe accuracy differed by -0.24 percentage points on average for paper-aligned SIGReg; its paired 95% interval includes zero. The paper-aligned arm had lower mean feature spread and covariance trace. This study does not measure the full ImageNet claim or establish isotropy across the representation.
- Notebook 03 encoder-probe accuracy was similar across the two arms. The paper-aligned RDMReg projector probe was higher by 6.94 percentage points on average, with a paired interval above zero. The teaching proxy was often nearly all zero; seed 7 had a zero projector and chance-level projector-probe accuracy. Its very low L0 and L1 values are not useful evidence of an effective sparse representation.
- The small five-seed intervals are uncertain. These are local measurements on fixed CIFAR-10 subsets and small CNNs. They do not validate the papers’ benchmark findings or prove broad superiority of either objective.
- All ten runs per notebook completed in the final outputs. The embedded checks passed. The final clean runs had no notebook errors and no scikit-learn convergence warnings. Model-run times below exclude notebook startup and dataset loading.

## Reproduction status

| Study | Local run time (sum of ten model-run timers) | Status |
| --- | ---: | --- |
| Notebook 02 LeJEPA objectives | 943.1 seconds | Completed local objective study; ImageNet-1K reproduction not run. |
| Notebook 03 Rectified LpJEPA objectives | 225.5 seconds | Completed local objective study; ImageNet-100 reproduction not run. |

Future exact benchmark work must select one paper result before execution and use the paper’s data, model scale, evaluation protocol, and baseline, or be labeled as a scaled or partial reproduction attempt. See the [reproduction matrix](../REPRODUCTION_MATRIX.md) and [evaluation protocol](../EVALUATION_PROTOCOL.md).
