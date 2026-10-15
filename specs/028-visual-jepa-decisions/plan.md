# Implementation Plan: Learned Visual JEPA Decision-Usefulness Lesson

**Branch**: `codex/action-conditioned-jepa-pilot` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a small visual action-conditioned JEPA lesson to the existing MountainCar planning path. Train compact models from locally generated rendered transitions. Compare the JEPA with an image-reconstructing latent model, a privileged state model, oracle dynamics, direct visual control, and policy-guided planning. Use CUDA for training and batched model rollouts when the installed PyTorch build supports the visible GPU. Keep the locked CPU path usable. Treat the study as a local small-scale experiment, not a paper reproduction.

## Technical Context

**Language/Version**: Python 3.10–3.13
**Primary Dependencies**: Existing project dependencies: PyTorch, NumPy, Gymnasium, Pillow, Matplotlib. No new dependency.
**Storage**: Generate transitions in memory. Keep small CSV evidence in `docs/evidence/`. Keep model weights, caches, and any larger outputs on E: or D: through the existing asset and uv path settings.
**Testing**: Standard-library unit tests for model shape, finite losses, collapse diagnostics, and CPU fallback; execute the notebook end-to-end; validate CSV pairing and record counts; run existing documentation and packaging checks requested for this project.
**Target Platform**: Python on Windows or Linux/WSL. CPU is supported. A compatible CUDA device is optional.
**Project Type**: Python package and Jupyter tutorials.
**Performance Goals**: Bound the study to generated MountainCar data, five training seeds for each learned model, 20 paired evaluation starts, and a fixed CEM budget. Use GPU batches for model training and planning when available. Report measured runtime and memory; set no unverified speed target.
**Constraints**: No downloaded data, pretrained checkpoint, robot hardware, or new package. Use the shared evaluation protocol. Preserve failed or missing runs. Keep state-privileged methods separate from image methods. A smaller run is exploratory or partial.

## Constitution Check

**Before design**: Pass. The scope is bounded, local, and evidence-led. The base CPU setup remains usable. The GPU path is optional and uses the project's existing CUDA wheel override. Paper findings and local measurements remain separate. The no-checkpoint and no-dataset decision avoids the compute and access requirements of released large models.

**After design**: Pass. The implementation adds one focused visual-dynamics module, one notebook, focused unit checks, and a small run report. It uses the existing SIGReg helper and evaluation rules. It does not alter the established notebook sequence or claim a paper reproduction.

## Design Decisions

1. Reuse Notebook 16's deterministic rendered MountainCar task and split its training, intervention, and evaluation seeds. Do not use evaluation cases during training or model selection.
2. Put compact model definitions and device selection in `src/jepa_tutorial/action_conditioned_visual.py`; keep data collection, protocol, training loops, interpretation, and plots visible in the notebook.
3. Form each visual observation from two consecutive frames so the model can infer velocity. Train a state-space MLP from simulator state as a privileged diagnostic, an image-reconstruction dynamics model with a future-frame decoder, and an action-conditioned JEPA that predicts the next two-frame context embedding with predictive loss plus the existing Epps–Pulley SIGReg regularizer. The JEPA has no pixel decoder; its output normalization has no learned affine scale. Flag collapse if embeddings are non-finite or more than half of validation dimensions have standard deviation below `1e-3`.
4. Score one-step predictions in their native spaces. Evaluate changed-action ranking and closed-loop success separately. Use the same candidate count, horizon, CEM update count, evaluation starts, and episode limit where methods support them. Mark observation and objective differences. The policy-guided JEPA condition uses a scripted visual-momentum action prior; it is not a learned policy.
5. Select CUDA only after a small tensor operation and synchronization succeed. Record PyTorch/CUDA versions, device, precision, training and evaluation time, and peak allocated GPU memory. Fall back to CPU if CUDA is absent or incompatible. Use float32 for this comparison.
6. Keep the five-seed comparison fixed. Pilot mode uses data/model seeds 1000/2021; smoke mode uses 9000/9021. A pre-gate exploratory run used 1100–1104 and 2121, 2127, 2139, 2153, and 2163 before the project-level prerequisite was noticed; exclude those groups permanently. Reserve final data seeds 5100–5104 and model seeds 7221, 7227, 7239, 7253, and 7263. A protocol-hash-matched validation-only pilot on the same device must pass before final model training, intervention collection, or episode evaluation starts. If it fails or is missing, full mode skips all final training and evaluation. Once the gate passes, preserve every planned seed, including failed or collapsed runs. A resource limit may stop the run, but does not lower the comparison threshold.
7. Keep the dependency-free intervention-record validator in `src/jepa_tutorial/evidence_validation.py`, separate from model primitives. Call it before the notebook writes intervention CSV files and unit-check its method, seed, pairing, status, and finite-metric rules.

## Project Structure

```text
src/jepa_tutorial/action_conditioned_visual.py
src/jepa_tutorial/evidence_validation.py
tests/test_action_conditioned_visual.py
notebooks/22_visual_jepa_decision_usefulness.ipynb
docs/reproductions/visual-jepa-decision-usefulness.md
docs/evidence/visual-jepa-model-runs.csv
docs/evidence/visual-jepa-interventions.csv
docs/evidence/visual-jepa-episodes.csv
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
docs/REPRODUCTION_MATRIX.md
README.md
specs/001-project-vision/alignment.md
specs/001-project-vision/remaining-work.md
specs/001-project-vision/tasks.md
```

The notebook is the learner-facing entry point. The model and device primitives stay in one module; the record validator is a small dependency-free helper that the notebook and focused unit tests share. There is no external API, so no contract directory is needed.

## Complexity Tracking

No constitution exception is required.
