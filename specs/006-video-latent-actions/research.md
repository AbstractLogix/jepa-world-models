# Research: Continuous Video-Derived Latent Actions

**Reviewed**: 2026-10-14

## Paper findings

Primary source: [Learning Latent Action World Models In The Wild, arXiv version 2](https://arxiv.org/html/2601.05230v2), revised 2026-01-20.

- The paper infers an action-like latent from current and future frame representations with an inverse dynamics model. A forward model predicts the next representation from past representations and that latent.
- It studies sparse continuous latents, noisy continuous latents, and vector-quantized latents. Figure 4 reports that sparse and noisy latents provide a wider range of prediction capacity, while vector quantization remains close to the deterministic baseline.
- The paper reports that latents transfer between different objects. It also finds that learned transformations are localized relative to the camera because the videos do not share a common embodiment.
- The reported setup uses a frozen V-JEPA 2-L frame encoder, a 128-dimensional latent action, and a ViT-L forward model. It trains on YoutubeTemporal-1B with 16-frame clips at 4 fps, 30,000 iterations, and batch size 1,024. Appendix A reports about 12 hours on 64 H100 GPUs.
- The paper does not link a code repository or released checkpoint. A first-party implementation or checkpoint was not confirmed in this source review. This is a review limit, not proof that none exists.

## Local design

Use a deterministic local renderer to create short clips of simple shapes under named camera transforms. Extract position and shape moments from each frame mask. This avoids external data, model weights, access gates, and long training. It makes camera and shape changes visible and repeatable.

Train a bounded continuous inverse dynamics model and a discrete code baseline on the same frame pairs. Use the inferred source action on matched target clips with a new shape or camera. Report held-out feature prediction error and transfer error over five training seeds. Keep generator motion labels out of model inputs and training loss.

The image moments are a fixed hand-built feature, not a learned JEPA representation. The synthetic clips have one foreground object, known camera transforms, and much less variation than natural videos. This is a tutorial analogue only. Its metrics do not match the paper's V-JEPA feature loss, LPIPS evaluation, or planning outcomes.

## Decisions

### Use generated clips

Use local generated clips instead of YoutubeTemporal-1B, Kinetics, RECON, or DROID. The paper uses a very large video dataset and reports a 64-H100 training run. This repository has no verified first-party checkpoint or confirmed data access for an exact reproduction. Generated clips make the lesson runnable on CPU and avoid redistributing external video.

### Use fixed image moments

Derive centroid and shape moments from rendered masks. This is simple to inspect and exposes the effect of a camera transform. It is a fixed feature extractor; do not describe it as a self-supervised or learned vision representation.

### Compare learned continuous and discrete codes

Use the same source clips, split, training budget, and seeds. Keep latent inference from frame pairs so learners can see the future-input boundary. Measure general prediction and transfer separately. Do not choose settings from held-out transfer results.

### Keep paper reproduction open

Before a paper-aligned attempt, confirm first-party code, checkpoint, data terms and access, the exact selected figure or table, the feature encoder, and a safe compute plan. A synthetic run cannot close these gates.

## Sources

- [Primary paper, arXiv version 2](https://arxiv.org/html/2601.05230v2)
- [arXiv record and version history](https://arxiv.org/abs/2601.05230)
