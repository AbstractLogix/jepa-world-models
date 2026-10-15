# Research: Point-Cloud JEPA Push-T Reproduction

**Reviewed**: 2026-10-14

## Claim selected

Re-run the Push-T rows for Point-LeWM and Point-Delta-JEPA from Table 2 of [Does Latent Planning Survive Point Clouds?](https://arxiv.org/abs/2608.29434). The paper reports a success rate of **83.6% ± 3.4** for Point-LeWM and **70.8% ± 7.7** for Point-Delta-JEPA. The difference is 12.8 points in favor of Point-LeWM. The paper reports mean and standard deviation across ten evaluation seeds, each with 50 episodes.

## Primary sources

| Asset or protocol | Pinned source | Evidence |
|---|---|---|
| Paper | [arXiv:2608.29434v2](https://arxiv.org/html/2608.29434v2) | Table 2 gives the two Push-T values. It uses identical fixed start-goal pairs and CEM planning. |
| Code | [fafraob/point-lewm](https://github.com/fafraob/point-lewm/tree/1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424) | Commit `1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424`, dated 2026-10-12. Its changed-file list includes image-evaluation changes but not `eval_lidar.py` or the Push-T LiDAR config. |
| Push-T v1 dataset | [Dataset files at v1 commit](https://huggingface.co/datasets/fafraob/point-cloud-pusht/tree/d3391d60371b6e9cbcd1c5737bdfc87b4af309ed) | Commit `d3391d60371b6e9cbcd1c5737bdfc87b4af309ed`; `pusht.lance.tar.zst`, 4,616,741,240 bytes; SHA-256 `80501fbca03fce36273be84086f3b0b6148b77fa4c5831d8085be7cdd91c1171`; CC-BY-4.0. |
| Push-T checkpoints | [Checkpoint files at model revision](https://huggingface.co/fafraob/point-cloud-pusht/tree/664278a4c0b4c5f607d0993744986b2d10b4168e) | Revision `664278a4c0b4c5f607d0993744986b2d10b4168e`; CC-BY-4.0. Point-LeWM `weights.pt`: 73,213,307 bytes, SHA-256 `42e1e85134497c8a2af0b9620d72c6421de767f1b85ee9778473c83292f81710`. Point-Delta-JEPA `weights.pt`: 83,012,963 bytes, SHA-256 `4a4315fdfae6b0ce15868fbb2bb4d80a6ebeda6c2d3f0a7c1f71cb8f1b36a237`. |
| Official protocol | [Official reproduction guide](https://github.com/fafraob/point-lewm/blob/1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424/docs/reproduction.md) | CEM uses 300 samples, 30 iterations, a five-step plan with frame skip 5, goal offset 25, a budget of 50 environment steps, and 50 episodes per seed. The ten seeds are `1, 25, 35, 36, 40, 42, 65, 80, 83, 99`. |
| Official environment | [Official installation guide](https://github.com/fafraob/point-lewm/tree/1b5a90ebf63e8ef165b1dfe100b81b12f4fa1424#installation) | Linux x86-64, CUDA 12, EGL, and 24 GB GPU memory for evaluation. The locked environment uses Python 3.10 and PyTorch 2.7 with CUDA 12.6. Building the point-cloud kernels takes about 20 minutes. |

## Decisions

### Use the v1 data and released checkpoints

The authors state that the released point-cloud checkpoints and paper numbers use v1. The October 2026 v2 release changes sensor-view images and fixes Push-T scan replay. Use v1 for a paper-number check. Do not substitute v2 even though it is the default dataset revision.

### Reproduce one matched comparison

Run only the Push-T Point-LeWM and Point-Delta-JEPA rows. The official protocol pairs them on the same ten seeds and 50 start-goal windows per seed. Do not expand this feature to the other environments, image baselines, target-to-latent tables, or model training.

### Report stochastic variation without overstating it

The point encoder resamples points during planning, so repeated runs can differ by a few points. Retain episode flags and seed-level rates. Compare each local mean with the paper's mean ± one reported standard deviation as a descriptive check only. Do not treat this as a significance test.

### Gate asset transfer on hardware

The official evaluation needs CUDA 12, EGL, and 24 GB GPU memory. The current host has an RTX 5070 with 12 GB VRAM, and the installed PyTorch is CPU-only. Record that blocker and do not transfer the 4.62 GB archive on this host.

## Alternatives considered

- **Use v2 data**: rejected because its scan replay differs from the checkpoint training data and paper result.
- **Re-train the models**: rejected because the selected claim is an evaluation result and the authors release checkpoints; training uses two A100 GPUs and a 72-hour budget per run.
- **Run all four environments and every baseline**: rejected for this feature because it broadens the claim and resource use beyond one matched result.
- **Use a CPU or 12 GB GPU fallback**: rejected because the official evaluator states a 24 GB CUDA requirement; a scaled run would not be the selected paper reproduction.

## Host status

The official resource requirement is documented. The current host does not meet it. The local evaluation is not started and remains open until a suitable host is available.
