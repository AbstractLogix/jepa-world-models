# Implementation Plan: Object-centric LeJEPA Lesson

**Branch**: `008-object-centric-lejepa`  
**Date**: 2026-10-14  
**Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/008-object-centric-lejepa/spec.md`

## Summary

Add a bounded comparison of image-level and mask-pooled LeJEPA objectives. Use the Oxford-IIIT Pet foreground trimaps and balanced breed classification as one downstream task. Use the same small encoder, data, probe, and training budget for both arms. Record the paper's current COCO result and keep the local transfer study separate from the paper benchmark.

## Technical Context

**Language/Version**: Python 3.10–3.13  
**Primary Dependencies**: Existing PyTorch, TorchVision, NumPy, SciPy, scikit-learn, and Matplotlib dependencies  
**Storage**: Downloaded Oxford-IIIT Pet files remain in `notebooks/data/`; no data or generated output is committed  
**Testing**: Execute the notebook in a clean kernel; inspect notebook JSON and run records  
**Target Platform**: CPU or CUDA-enabled local machine; default run uses five seeds  
**Project Type**: Jupyter tutorial repository  
**Performance Goals**: Complete the bounded five-seed study on the stated small dataset without requiring paper-scale hardware  
**Constraints**: No object-centric first-party code was linked by the paper's current v2 source review. Use an explicit reduced implementation. Do not call it a reproduction of the COCO benchmark. No new dependency.  
**Scale/Scope**: 37 classes; fixed, balanced subset of 16 trainval images per class; full official test split; 96-pixel input; small CNN; 5 epochs per seed and method; 5 seeds

## Constitution Check

- **Teach from evidence**: Pass. Link the latest paper and the official dataset source. Separate paper values from local values.
- **Label teaching examples honestly**: Pass. Label the local work as an exploratory transfer study with a reduced objective. Keep the paper benchmark not run.
- **Make the learning path usable**: Pass. The notebook uses public data, needs no robot or private source, and records its download size and license.
- **Measure before making performance claims**: Pass. Use a fixed held-out split, one balanced-accuracy probe, five seeds, and full run records.
- **Keep changes small and reviewable**: Pass. Add one notebook and one report; reuse current loss code and evaluation protocol.
- **No principle conflict**: None.

## Design

1. Load the official `trainval` and `test` splits with category and segmentation targets.
2. Build a fixed, balanced `trainval` subset using one data seed. Convert pet and outline trimap values into the foreground mask; exclude examples whose resized trimap has no foreground before balanced sampling, and record the count.
3. Train two matched arms. The image arm aligns projections pooled across the full feature map. The object arm aligns projections pooled over the transformed foreground mask. Both use the current Epps–Pulley SIGReg helper and the same reduced CNN and two-view augmentation.
4. Do not add the paper's instance projection loss or copy-paste recipe. Oxford-IIIT Pet has one labeled foreground pet per image, so this study cannot check co-occurring instance separation. State this limit beside the local result.
5. Extract frozen feature maps for the same trainval and full official test split. Pool probe features over the same supplied pet masks for both arms; when a test trimap has no foreground, use global-average pooling for both methods and report the fallback count. Fit one balanced logistic-regression probe per seed and report balanced accuracy.
6. Record raw scores, seed-level paired differences, mean, median, 95% Student-t interval, runtime, device, and software versions.
7. Update the lesson map, roadmap, reproduction matrix, project feature tasks, and alignment records.

The target paper's Table 1 reports ADE20k object-classification balanced accuracy 0.555 for its semantic object representation and 0.426 for image-level LeJEPA. The local Oxford-IIIT Pet result uses a different dataset, smaller CNN, reduced loss, and fewer epochs. It tests transfer of the comparison idea only.

## Project Structure

### Documentation (this feature)

```text
specs/008-object-centric-lejepa/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── tasks.md
└── checklists/requirements.md
```

### Source Code

```text
notebooks/13_object_centric_lejepa.ipynb
docs/reproductions/object-centric-lejepa.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
docs/REPRODUCTION_MATRIX.md
specs/001-project-vision/{tasks.md,alignment.md,remaining-work.md}
```

**Structure Decision**: Keep the study in one notebook, following the existing tutorial layout. Reuse `src/jepa_tutorial/paper_losses.py`; add no shared abstraction or dependency.

## Constitution Check After Design

Pass. The design states the human mask source, label use, reduced objective, dataset differences, evaluation protocol, and exact benchmark status. The one-object-per-image dataset is not suitable for validating the paper's instance-separation result.

## Complexity Tracking

No constitution violations.
