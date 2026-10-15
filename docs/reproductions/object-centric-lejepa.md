# Object-centric LeJEPA: source review and local study

**Reviewed**: 2026-10-14

**Status**: Five-seed Oxford-IIIT Pet study complete; exact paper benchmark not run

## Paper finding

The current source is v2 of [Mask-supervised Object-centric Representation Learning with LeJEPA](https://arxiv.org/abs/2607.02404), dated 2026-10-05. Its main COCO experiment uses human ground-truth instance masks. SAM 2.1 masks and Felzenszwalb–Huttenlocher superpixels are separate mask-source ablations. The method combines object-level semantic alignment with SIGReg, an instance-separation loss, and copy-paste augmentation.

Table 1 reports ADE20k object-classification balanced accuracy of 0.555 for the paper's semantic object representation and 0.426 for image-level LeJEPA. It reports DAVIS tracking J&F of 0.683 and 0.626. The paper says each configuration uses one training seed. Its reproduction section says object-centric training and SAM-mask code will be released, but v2 does not link a first-party repository. The separate [official image-level LeJEPA repository](https://github.com/galilai-group/lejepa) is available; it is not the implementation of this extension.

## Local study

[Notebook 13](../../notebooks/13_object_centric_lejepa.ipynb) compares image-level and mask-pooled alignment on Oxford-IIIT Pet. It uses the official trainval and test splits, a fixed subset of 16 trainval images per breed, human trimaps, a small CNN, five seeds, five epochs, and a frozen balanced breed-classification probe. Trimap values 1 and 3 form the foreground. Breed labels do not enter the self-supervised loss. Both methods use the same images, augmentations, model, optimizer, and probe.

The image archive has 791,918,971 bytes and MD5 `5c4f3ee8e5d25df40f4fd59a7f44e54c`. The annotations archive has 19,173,078 bytes and MD5 `95a8c909bbe2e81eed6a22bccdf3f68f`. The archives loaded as 3,680 trainval and 3,669 test images across 37 classes.

Seven trainval trimaps contain no pet or outline pixels. The notebook excludes them before balanced sampling and records the count. The fixed 592-image subset has 16 valid masks per class; its index SHA-256 is `44ec239cff63e8b963b81a8632c81126a97e452db01bffd7fc96eebce6f26f0d`. All 3,669 test images remain in the evaluation. Seven test trimaps have no foreground; both methods use global-average pooling for those examples and record the fallback count.

## Execution record

The final notebook run used a clean kernel and completed all 10 model runs with no notebook errors. The run took 984.3 seconds including notebook startup; the study cell reported 962.8 seconds. Model training took 49.4–52.1 seconds per run, and feature extraction took 21.1–22.9 seconds per run.

| Item | Value |
| --- | --- |
| Python / PyTorch / TorchVision | 3.13.14 / 2.10.0+cpu / 0.25.0+cpu |
| NumPy / SciPy / scikit-learn | 2.5.4 / 1.18.1 / 1.9.1 |
| Host | AMD Ryzen 9 5900X, 24 logical CPUs, 31 GiB RAM; NVIDIA RTX 5070 with 12,227 MiB, unavailable to this CPU-only PyTorch environment |
| Model | Four convolution stages (32, 64, 128, 256 channels); 388,896 encoder parameters. The projection head has 279,360 parameters; 668,256 parameters train per method. |
| Input and training | 96×96 images, batch size 64, five epochs, SIGReg with 128 projections and 17 points |
| Optimizer | AdamW, learning rate 5e-4, weight decay 5e-4; gradient norm clipped to 1.0 |
| Seeds | 7, 17, 27, 37, 47 for each method |
| Test split | All 3,669 official test images; 7 mask examples use the shared global-pooling fallback |
| Run status | 10 of 10 complete; no non-finite results or failed runs |

The metrics below are balanced accuracy on the official test split. The intervals are 95% Student-t intervals across five training seeds. They are uncertain at this sample size.

| Method | Feature pooling | Seeds | Mean | Median | 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Image-level objective | Global average | 5 | 0.0625 | 0.0593 | [0.0521, 0.0728] |
| Image-level objective | Pet mask with global fallback | 5 | 0.1053 | 0.1038 | [0.0951, 0.1154] |
| Object-level objective | Global average | 5 | 0.0698 | 0.0727 | [0.0624, 0.0773] |
| Object-level objective | Pet mask with global fallback | 5 | 0.1139 | 0.1151 | [0.1025, 0.1252] |

| Seed | Image global | Object global | Image pet-mask/fallback | Object pet-mask/fallback |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.0521 | 0.0759 | 0.0962 | 0.1211 |
| 17 | 0.0589 | 0.0610 | 0.1038 | 0.0986 |
| 27 | 0.0593 | 0.0729 | 0.0996 | 0.1207 |
| 37 | 0.0710 | 0.0666 | 0.1102 | 0.1151 |
| 47 | 0.0710 | 0.0727 | 0.1165 | 0.1138 |

For pet-mask/fallback features, the paired object-minus-image difference had mean 0.0086, median 0.0049, and 95% interval [-0.0084, 0.0256] across the five matched seeds. The interval includes zero. This small local study does not establish a reliable advantage for either training objective.

The first complete-data run failed because the original training subset included empty masks. Inspection found seven invalid masks in each official split. The corrected notebook excludes empty trainval masks before subset selection and applies the same global-pooling fallback to both methods on the seven affected test images. A focused regression check confirmed both behaviors before the final run.

## Interpretation and limits

This is an exploratory transfer study, not a reproduction of the paper's result. It changes the data, encoder, image size, training budget, objective, and downstream task. It also omits the paper's instance-separation loss and copy-paste augmentation. Oxford-IIIT Pet has one main pet per image, so the study cannot test within-image instance separation. Its scores do not validate or refute the COCO, ADE20k, DAVIS, NAVI, or ImageNet findings.

The Oxford-IIIT Pet dataset is about 800 MB and is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Image copyright remains with the original owners. The downloaded files and executed notebook remain outside version control.

## Remaining work

The exact paper reproduction remains open. It needs the authors' object-centric implementation, COCO masks and training data, suitable compute, and the paper-matched ADE20k or DAVIS evaluation. Recheck the source and artifact status before planning that work.
