# Implementation Plan: Continuous Video-Derived Latent Actions

**Branch**: `006-video-latent-actions` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from project vision task T013.

## Summary

Add notebook 12 with generated grayscale clips, a fixed frame-moment feature extractor, and a comparison between bounded continuous latent actions and discrete codes. Measure one-step prediction and transfer across object shape and camera transforms. Record CPU results and align the curriculum, roadmap, paper-reproduction matrix, and project task. This is not a paper reproduction.

## Technical Context

**Language/Version**: Python 3.10–3.13.

**Primary Dependencies**: Existing NumPy, PyTorch, and Matplotlib dependencies. No new dependency.

**Storage**: Generate clips in memory. Keep notebook outputs, model weights, and data outside tracked paths.

**Testing**: Parse notebook JSON and code cells. Run notebook 12 from top to bottom on CPU. Check clip validity, disjoint splits, finite latents and predictions, complete five-seed output, and transfer metrics.

**Target Platform**: Local Jupyter on CPU; CUDA is optional.

**Project Type**: Educational notebook repository.

**Performance Goals**: Generate small 32×32 clips and complete five seeds for each of two methods within a short local CPU run.

**Constraints**: Use generated shapes and views only. Extract fixed image moments from rendered frame masks. Do not use generator action labels as model inputs. Do not claim learned visual features, real-video generalization, robot performance, or paper reproduction.

**Scale/Scope**: One notebook, one local result report, and targeted updates to curriculum, roadmap, reproduction matrix, and project tracking. No public API or shared module.

## Constitution Check

- Teach from evidence: pass. The notebook links the primary version 2 paper and reports its method and scale.
- Label teaching examples honestly: pass. The experiment is marked as a synthetic analogue with a fixed hand-built feature extractor.
- Make the learning path usable: pass. Generated data, CPU use, and expected outputs are stated.
- Measure before making performance claims: pass. Methods use the same data and seeds; report raw runs and uncertain intervals.
- Keep changes small and reviewable: pass. Add one notebook and one evidence report; add no dependencies or shared code.
- Data and model terms: pass. No external data or checkpoints are downloaded.

## Research Summary

See [research.md](research.md). The paper reports that continuous sparse or noisy latents support a wider prediction-capacity range than vector quantization, and that learned actions can transfer between objects while remaining camera-relative. Its training scale and data are too large for this tutorial. The local notebook uses generated clips and fixed frame moments to teach and measure a small analogue.

## Design and Interfaces

See [data-model.md](data-model.md). Generate clips of circles and squares for training, plus a triangle shape for held-out shape transfer. Train on one camera view and evaluate held-out motions in the same view, a new shape, and a rotated and zoomed camera. Each sample is a matched transition with the same generated world motion across views.

The fixed frame feature contains a foreground centroid and second moments measured from a rendered mask. Reject out-of-frame objects with a clear error before training. Train an inverse model to infer either a bounded two-dimensional continuous action or one of eight discrete codes from adjacent features. Train a forward model to predict next-frame features from the current feature and inferred action. Use an MSE prediction loss, a small continuous-action regularizer, and a variance floor to discourage action collapse. Use data seed `20261014` and initialization seeds `[7, 17, 27, 37, 47]` for both methods, with identical training transitions, batch order, and epoch count.

For transfer, infer an action from a source transition in the training view and apply that action to the target's current feature. Compare the predicted target feature with the matched target next feature. Report same-view held-out error, new-shape error, and new-camera error separately. Do not use held-out data to tune the model.

There is no external interface or contract. The notebook is the learner-facing interface. Report model runs and figures in [docs/reproductions/latent-actions.md](../../docs/reproductions/latent-actions.md); do not commit generated clips or model outputs.

## Project Structure

```text
notebooks/12_video_latent_actions.ipynb
docs/reproductions/latent-actions.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
docs/REPRODUCTION_MATRIX.md
specs/001-project-vision/tasks.md
specs/006-video-latent-actions/
```

**Structure Decision**: Keep the lesson in the numbered notebook series. Do not add a shared package module or dependencies for one teaching example.

## Risks and Mitigations

- The fixed image moments reveal object location by design. Explain that this makes the experiment easier than learning visual features from natural video.
- A camera change can alter the pixel-space action direction or scale. Report camera transfer separately.
- Small continuous latent actions can collapse to zero. Bound them and use a documented spread constraint; show their range and standard deviation.
- The inverse model observes a future frame. Explain that this is training-time latent inference, not a deployable action controller.
- Small generated data can make a discrete codebook look better or worse by chance. Report five seeds and all runs, with uncertain intervals.

## Constitution Check After Design

Pass. The design keeps the local result bounded, repeatable, and separate from the paper's in-the-wild results. The notebook does not claim real camera, embodiment, or robot transfer.
