# Implementation Plan: Reproducible Legacy Lesson Evidence

**Branch**: `009-reproducible-legacy-lessons` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification in `specs/009-reproducible-legacy-lessons/spec.md`

## Summary

Add fixed random seeds to the existing synthetic and control lessons. Make the two CartPole planner comparisons use the same fixed initial-state seeds. Run notebooks 01 and 04–08 from clean CPU kernels, record local results and limits, and align the coverage and project status documents. Keep all outputs and checkpoints outside version control.

## Technical Context

**Language/Version**: Python 3.10–3.13  
**Primary Dependencies**: Existing PyTorch, TorchVision, Gymnasium, timm, NumPy, pandas, Matplotlib, scikit-learn, Jupyter, and nbclient dependencies  
**Storage**: Source data and reusable checkpoints stay in ignored `notebooks/data/` and `notebooks/checkpoints/`; executed notebook copies stay under `/tmp`  
**Testing**: Parse notebook JSON and compile code cells; execute each notebook in a clean CPU kernel; repeat notebooks 05 and 06 to verify matched episode results  
**Target Platform**: Linux CPU; GPU is not required for this feature  
**Project Type**: Jupyter tutorial repository  
**Performance Goals**: Complete the six bounded notebook runs on the declared CPU environment without changing their lesson scope  
**Constraints**: Add no dependency; preserve current notebook order; keep toy and paper claims separate; do not commit data, weights, checkpoints, or execution outputs  
**Scale/Scope**: Six notebooks, 20–25 evaluation episodes for each CartPole method, one training seed for the existing ablation grid

## Constitution Check

- **Teach from evidence**: Pass. The report records measured local output and links official NumPy and Gymnasium API documentation.
- **Label teaching examples honestly**: Pass. Existing toy targets, small models, and single-seed ablations stay illustrative or exploratory.
- **Make the learning path usable**: Pass. The run uses CPU and public data or weights; first-run downloads remain documented.
- **Measure before making performance claims**: Pass. Planning methods use the same fixed episodes and report raw returns. The ablation remains explicitly single-seed.
- **Keep changes small and reviewable**: Pass. Seed and evaluation code changes stay in existing notebook cells; no shared runtime abstraction or dependency is added.

## Design

1. Replace the general eigenvalue solver in notebook 01's real symmetric covariance plot with `numpy.linalg.eigh`. Set the toy model seed before initialization and training.
2. Keep notebook 04's fixed CIFAR subset and model seed. Pin the timm pretrained feature source to its reviewed revision, use ImageNet normalization, fit the projector and target normalization on the training split only, evaluate descriptor alignment and retrieval on held-out images, record the weight license, and label the task as an analogue.
3. Seed notebook 05's random CartPole data and evaluate its random and planned policies on the same 25 initial-state seeds. Seed both random actions and random planner candidates. Score the same sampled candidates as a batch to keep CPU runtime bounded.
4. Seed notebook 06's planner evaluation and use the same 20 initial-state seeds for random shooting and gradient planning. Reset its candidate RNG for each episode and score random-shooting candidates as a batch.
5. Seed notebook 07's action space and every environment reset. Label its current one-step score as training-set evaluation.
6. Seed notebook 08's collected data and each ablation model. Reuse the same 10 reset seeds and candidate RNG seed for every configuration. Score the same sampled continuations in a batch. Mark the eight single-seed outcomes exploratory.
7. Execute the notebooks in isolated CPU kernels. Store executed copies in `/tmp` and keep the reusable notebook 05 checkpoint in the existing ignored checkpoint path. Add one report with software versions, device, run times, raw results, failures, and limitations.
8. Update curriculum coverage, project alignment, and remaining work to match the run evidence.

## Project Structure

### Documentation (this feature)

```text
specs/009-reproducible-legacy-lessons/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/requirements.md
└── tasks.md
```

### Source Code

```text
notebooks/01_intuition.ipynb
notebooks/04_vjepa_plus_v1_jepa.ipynb
notebooks/05_value_guided_planning.ipynb
notebooks/06_gradient_based_planning.ipynb
notebooks/07_latent_action_world_models.ipynb
notebooks/08_physical_planning_ablation.ipynb
docs/evidence/legacy-lessons-local-run.md
docs/CURRICULUM_COVERAGE.md
specs/001-project-vision/{tasks.md,alignment.md,remaining-work.md}
```

**Structure Decision**: Keep lesson logic in the existing notebooks. Store only compact measured evidence in `docs/evidence/`; keep executed notebooks, source datasets, and checkpoints in temporary or ignored storage.

## Constitution Check After Design

Pass. The design fixes seed sources and paired evaluation cases, records actual local evidence, and keeps toy results separate from paper findings. The single-seed ablation is clearly exploratory.

## Complexity Tracking

No constitution violations.
