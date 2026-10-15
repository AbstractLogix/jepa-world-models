# Data Model: Continuous Video-Derived Latent Actions

## Generated clip

- **Fields**: clip ID, split, shape, camera transform, seed, and ordered grayscale frames.
- **Rules**: Use data seed `20261014`. Render one object on a fixed background. Use circles and squares for training, a triangle for shape-transfer evaluation, and a distinct camera transform for camera-transfer evaluation. Keep motions away from frame boundaries or count invalid pairs.
- **Source**: In-memory local generator. No external video is loaded.

## Frame feature

- **Fields**: normalized x/y centroid and second moments from the foreground mask.
- **Rules**: Compute from each rendered frame using a fixed intensity threshold. Fail if the foreground mask is empty. The feature is a hand-built image summary, not a learned encoder.

## Transition sample

- **Fields**: source clip ID, time index, current and next frame features, matched world-motion key.
- **Rules**: Split by clip ID. Generator motion labels are available only to construct matched transfer cases and are not inputs to inverse or forward models.

## Latent-action run

- **Fields**: method (`continuous` or `discrete`), training seed, inferred actions, forward predictions, settings, losses, code or action usage.
- **Rules**: Both methods use the same training pairs and fixed evaluation sets. Use training seeds `[7, 17, 27, 37, 47]`. Continuous actions are bounded to [-1, 1]. Discrete actions use eight codes. A failed or non-finite run stays visible and does not enter aggregate metrics.

## Transfer case

- **Fields**: matched source and target transitions, transfer type (`same-view`, `new-shape`, or `new-camera`), predicted next feature, true next feature, error.
- **Rules**: Infer an action from the source transition and apply it to the target's current feature. Report each transfer type separately.
