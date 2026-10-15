# Local Run: Continuous Video-Derived Latent Actions

**Run date**: 2026-10-14

**Claim status**: Locally measured synthetic analogue. This is not a paper reproduction, real-video transfer result, or robot result.

## Paper finding selected for future reproduction

The version 2 paper reports in Figure 4 that sparse and noisy continuous latent actions span a wider range of one-step prediction capacity than vector-quantized actions. It also reports that learned actions can transfer across objects while remaining mainly localized relative to the camera. These are paper-reported findings from its own setup.

The paper uses a frozen V-JEPA 2-L frame encoder, a ViT-L forward model, and default 128-dimensional actions. It trains on YoutubeTemporal-1B with 16-frame clips at 4 fps, 30,000 iterations, and batch size 1,024. Appendix A reports about 12 hours on 64 H100 GPUs. See the [primary paper, version 2](https://arxiv.org/html/2601.05230v2).

## Local method

This notebook renders simple grayscale clips in memory. Training uses 1,024 clips of circles and squares from the identity camera. The held-out set has 256 new clips. Two transfer sets have 128 matched motions each: a triangle under the identity camera and a circle under a rotated and zoomed camera. Each clip has three frames.

A fixed threshold extracts normalized centroid x/y, variance x/y, and covariance xy from each frame. This is a hand-built image feature, not a learned visual encoder. Generator motion values are used to match transfer examples; the models do not receive them as input or training targets.

The continuous method uses a two-dimensional tanh-bounded inverse model with a small L1 and variance-floor penalty. The discrete method uses eight hard Gumbel-Softmax codes with a mild usage-entropy term. Both use a 64-unit forward MLP, MSE loss, Adam at 0.001, batch size 128, and 20 epochs. The comparison uses data seed 20261014 and training seeds 7, 17, 27, 37, and 47.

An inferred action is measured on same-view held-out clips, then applied to matched new-shape and new-camera clips. The metric is mean squared error across normalized frame-feature dimensions and held-out cases. The 95% Student-t intervals use four degrees of freedom and five training seeds; they are uncertain.

## Environment and result

| Item | Observed value |
| --- | --- |
| Runtime | Python 3.13.14, PyTorch 2.10.0+cpu, NumPy 2.5.4, Matplotlib 3.11.2 |
| Host | Linux WSL2, CPU; CUDA unavailable for this run |
| Training/evaluation | 1,024 training clips, 256 same-view held-out clips, 128 cases per transfer setting |
| Runs | 5 seeds per method; 10 completed, 0 failed |
| Training and inference time | 2.23 seconds total for the 10 runs |
| Downloads | None |

### Per-seed normalized feature MSE

| Method | Seed | Same view | New shape | New camera |
| --- | ---: | ---: | ---: | ---: |
| Continuous | 7 | 0.10868 | 0.51552 | 0.43894 |
| Continuous | 17 | 0.21855 | 0.35246 | 0.20035 |
| Continuous | 27 | 0.14356 | 0.43118 | 0.40040 |
| Continuous | 37 | 0.15018 | 0.44633 | 0.49556 |
| Continuous | 47 | 0.14995 | 0.55368 | 0.57711 |
| Discrete | 7 | 0.22652 | 0.40219 | 0.22731 |
| Discrete | 17 | 0.21628 | 0.40838 | 0.29428 |
| Discrete | 27 | 0.19703 | 0.43021 | 0.29747 |
| Discrete | 37 | 0.22544 | 0.34751 | 0.22422 |
| Discrete | 47 | 0.21459 | 0.34207 | 0.27828 |

### Mean, variation, and uncertain 95% interval across seeds

| Method | Setting | Mean | Median | SD | 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Continuous | Same view | 0.15418 | 0.14995 | 0.03988 | [0.10468, 0.20369] |
| Continuous | New shape | 0.45983 | 0.44633 | 0.07817 | [0.36279, 0.55687] |
| Continuous | New camera | 0.42247 | 0.43894 | 0.14085 | [0.24761, 0.59733] |
| Discrete | Same view | 0.21597 | 0.21628 | 0.01185 | [0.20126, 0.23068] |
| Discrete | New shape | 0.38607 | 0.40219 | 0.03915 | [0.33748, 0.43467] |
| Discrete | New camera | 0.26431 | 0.27828 | 0.03595 | [0.21969, 0.30894] |

Continuous actions had lower mean same-view error in this run. Discrete actions had lower mean error in both transfer settings. The five-seed intervals are uncertain and overlap for the transfer settings. The result does not show a general advantage for either representation.

## Limits and next reproduction gates

- The generator has one foreground object, a fixed background, two known camera transforms, and simple shapes. It does not represent natural video or varied real embodiments.
- The features directly expose object position. There is no learned visual encoder, V-JEPA checkpoint, video decoder, or paper-scale forward model.
- The inverse model uses the future frame to infer a latent action. This is an explanatory training-time step, not a deployable controller.
- The errors are not the paper's latent-feature prediction error, LPIPS, or planning success metric.
- The paper source review did not confirm first-party code, a released checkpoint, or usable data access. This source-review limit does not prove those resources do not exist.
- An exact Figure 4 reproduction remains not run. The [paper-reproduction matrix](../REPRODUCTION_MATRIX.md) records the required code, checkpoint, data, evaluation, and compute gates.

This report records a local teaching measurement only. It does not validate the paper's findings.
