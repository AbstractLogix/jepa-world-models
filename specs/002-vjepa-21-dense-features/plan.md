# Implementation Plan: V-JEPA 2.1 Dense Video Features Tutorial

**Branch**: `002-vjepa-21-dense-features` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: [spec.md](spec.md)

## Summary

Add an optional ninth notebook that loads the smallest official V-JEPA 2.1 checkpoint from a pinned upstream revision, creates a deterministic 16-frame clip locally, visualizes its dense patch features, and measures one known-motion correspondence against a pixel baseline. Keep checkpoint download opt-in. Use the public model entrypoint's 384-pixel input and label the synthetic task as an illustration, not a paper reproduction.

## Technical Context

**Language/Version**: Python 3.10–3.13; use Python 3.12 for the recorded local run.

**Primary Dependencies**: Existing NumPy, Matplotlib, Pillow, timm, and PyTorch installation; add the `vjepa21` extra for einops and headless OpenCV required by the pinned Hub preprocessing import. Install PyTorch separately with `requirements-torch.txt`.

**Storage**: Generated clip and cached checkpoint under ignored `notebooks/checkpoints/`; no generated output is committed.

**Testing**: Validate notebook JSON and run all no-download cells. With the explicit opt-in flag, execute the notebook with the official checkpoint on CPU and record local resource measurements.

**Target Platform**: Local Linux, Windows, or macOS Python environment; CPU path is required, accelerator path is optional.

**Project Type**: Educational notebook repository.

**Performance Goals**: No fixed runtime promise. Report actual wall time, peak process memory when available, device, and input shape.

**Constraints**: The 1.66 GB checkpoint requires network access and local storage. No robot hardware, external video, or dataset is required. The public entrypoint uses 384-pixel inputs; the notebook reduces the temporal input to 16 frames.

**Scale/Scope**: One notebook, one optional dependency extra, and aligned curriculum and research documentation.

## Constitution Check

- **Teach from evidence**: Pass. Pin the official code revision and checkpoint; keep paper statements separate from local measurements.
- **Label examples honestly**: Pass. The generated clip is a diagnostic illustration; the lesson is not a paper reproduction.
- **Make the path usable**: Pass. Clip generation and visualization run without downloads; model loading is explicit.
- **Measure comparisons**: Pass. Name the pixel baseline, correspondence metric, input, preprocessing, device, and run time.
- **Keep changes small**: Pass. Add one notebook and two opt-in runtime dependencies required by the official loader and preprocessing, without changing existing lesson order or code layout.

## Design

The notebook uses the official Hub entrypoint at a fixed Git commit with `pretrained=False`, the public 384-pixel input size, and 16 frames. It downloads the official checkpoint directly only after the learner sets an environment opt-in, then strictly loads the official `ema_encoder` weights. This avoids the pinned repository revision's default checkpoint URL, which points to localhost. The official preprocessing function uses a 384-pixel center crop; the notebook records its resulting tensor shape. The notebook saves no results automatically.

The correspondence exercise compares cosine similarity between normalized patch vectors for the moving object with a mean-RGB patch baseline. Both report the retrieved patch, target distance, and similarity. No pass threshold is used.

## Project Structure

```text
notebooks/09_vjepa21_dense_features.ipynb
pyproject.toml
poetry.lock
README.md
CURRICULUM_GUIDE.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
docs/evidence/vjepa21-local-run.md
specs/002-vjepa-21-dense-features/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

**Structure Decision**: Use the existing numbered notebook curriculum and document its optional-model boundary in the existing roadmap and coverage map. No new Python package or public API is needed.

## Post-Design Constitution Check

Pass. The design adds an optional model extra, preserves the separate PyTorch installation, records source and terms uncertainty, and limits its claim to a measured synthetic-clip demonstration.
