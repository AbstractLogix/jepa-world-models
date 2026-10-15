# Implementation Plan: VL-JEPA Real Label Embeddings

**Branch**: `005-vljepa-real-labels` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: [Project vision task T012](../001-project-vision/tasks.md)

## Summary

Add one advanced notebook that predicts fixed embeddings of real CIFAR-10 class prompts from frozen ImageNet ResNet-18 features. Compare the resulting text-prompt ranking with a linear classifier that uses the same features and data. Record the local results and align the roadmap and curriculum map. This is a small supervised analogue, not a VL-JEPA reproduction.

## Technical Context

**Language/Version**: Python 3.10–3.13

**Primary Dependencies**: PyTorch and TorchVision from `requirements-torch.txt`; optional `transformers>=5,<6` and `safetensors>=0.4,<1`; existing NumPy, scikit-learn, and Matplotlib.

**Storage**: CIFAR-10 and model files use the existing ignored `notebooks/data/` and framework caches. No checkpoint or generated output is committed.

**Testing**: Parse notebook JSON and run `notebooks/11_vl_jepa.ipynb` from top to bottom on CPU with all downloads cached. Check sample balance, disjoint official splits, finite embeddings and scores, and complete five-seed output.

**Target Platform**: Local Jupyter on Python 3.10–3.13; CPU is the required path and CUDA is optional.

**Project Type**: Educational notebook repository.

**Performance Goals**: Use 100 training and 100 held-out images per class, five fixed seeds, and small linear/MLP heads. Keep full image inference to 2,000 CIFAR-10 images.

**Constraints**: First run needs network access and about 300 MB for CIFAR-10, ResNet-18, and MiniLM files. Do not silently change model revision or dataset. State the CIFAR-10 and TorchVision weight license limits. Do not present local metrics as paper results.

**Scale/Scope**: One notebook, one result report, one optional dependency extra, and targeted curriculum, roadmap, and project-task updates.

## Constitution Check

- Teach from evidence: pass. Use the primary VL-JEPA paper, official CIFAR-10 page, official TorchVision model source, and pinned text model card.
- Label examples honestly: pass. Call the lesson a supervised analogue, not a reproduction, zero-shot result, or open-vocabulary evaluation.
- Make the learning path usable: pass. Give installation, download, and CPU/GPU details before the run.
- Measure before making performance claims: pass. Use the same frozen features and fixed balanced splits for the baseline and predictor; report five seeds and all raw results.
- Keep changes small and reviewable: pass. Add one optional dependency extra and one new notebook, then update only linked project records.
- Data and model terms: pass with limits. Record the CIFAR-10 page's missing license statement and TorchVision's notice that weights can carry separate terms. Do not redistribute either.

## Research Summary

See [research.md](research.md). VL-JEPA maps images to visual embeddings, predicts a real target-text embedding, and compares in a shared latent space. The paper uses V-JEPA 2 ViT-L, a Llama-3.2-based predictor, EmbeddingGemma, video/image-caption data at billions-of-samples scale, and video classification and retrieval benchmarks. This lesson uses CIFAR-10 labels, a frozen ImageNet ResNet-18, a frozen MiniLM encoder, and a small predictor. The shared idea is embedding prediction; the architecture, data, scale, and claims differ.

## Design and Interfaces

See [data-model.md](data-model.md) for the sample, prompt, run, and metric records. The notebook is the learner-facing interface; there is no public service or API.

Use one fixed balanced subset: 100 examples per class from CIFAR-10's official training split and 100 examples per class from its official test split. Derive sample selection from one disclosed data seed. Extract frozen 512-dimensional image features once. Encode one fixed prompt per class with the pinned `sentence-transformers/all-MiniLM-L6-v2` revision and the documented attention-mask mean pooling and L2 normalization. Train a small image-to-embedding predictor with cosine loss and a `Linear(512, 10)` classifier with cross-entropy. Use the same five seeds, training data, test data, and cached image features for each method. Rank predicted text embeddings against the ten normalized prompt embeddings by cosine similarity. Report individual scores, mean, median, and an uncertain 95% t interval across five training runs; also report a paired per-image difference with a class-stratified bootstrap interval for the fixed test sample.

Pin `ResNet18_Weights.IMAGENET1K_V1`, use its supplied preprocessing, and record the Python, PyTorch, TorchVision, Transformers, CPU/GPU, data seed, training seeds, head settings, and per-seed results. Keep network access and first-run download sizes visible. Do not write cache files into tracked paths.

## Project Structure

```text
notebooks/11_vl_jepa.ipynb
docs/
├── CURRICULUM_COVERAGE.md
├── ROADMAP.md
├── REPRODUCTION_MATRIX.md
└── reproductions/vl-jepa.md
specs/
└── 005-vljepa-real-labels/
    ├── spec.md
    ├── plan.md
    ├── research.md
    ├── data-model.md
    ├── quickstart.md
    ├── tasks.md
    └── checklists/requirements.md
```

**Structure Decision**: Follow the numbered notebook curriculum and keep results with the other reproduction records. Do not add a shared package module for a single lesson.

## Risks and Mitigations

- CIFAR-10 labels are narrow and have no within-class captions. State that this is closed-set supervised classification, not zero-shot or open-vocabulary evaluation.
- A pretrained ImageNet classifier backbone can transfer class information. Keep it frozen and label the ImageNet supervision; this test does not isolate the text objective from visual pretraining.
- A two-layer predictor and linear classifier have different output dimensions and objectives. Describe the comparison as a baseline comparison, not a controlled capacity study.
- CIFAR-10 and the ImageNet weights have terms that are not fully stated by their source pages. Record the limits and do not redistribute assets.
- First-run downloads can fail or change upstream. Pin the MiniLM revision, use the named TorchVision weight enum, fail clearly on missing data/model files, and record installed versions.

## Constitution Check After Design

Pass. The design identifies the exact local task and baseline, makes all downloads and limits visible, keeps the reproduction claim fail-closed, and adds only dependencies needed for text embeddings.
