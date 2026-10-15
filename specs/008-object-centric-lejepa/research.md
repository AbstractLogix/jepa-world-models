# Research: Object-centric LeJEPA Lesson

**Reviewed**: 2026-10-14

## Primary paper

The current version is v2 of [Mask-supervised Object-centric Representation Learning with LeJEPA](https://arxiv.org/abs/2607.02404), dated 2026-10-05. The v2 abstract and methods say that the main COCO experiment uses human ground-truth instance masks. SAM 2.1 proposals and Felzenszwalb–Huttenlocher superpixels are separate mask-source ablations. The earlier v1 abstract described SAM proposals as the main mask source, so the roadmap must reflect v2.

The method has two feature spaces. Its semantic object space aligns representations of the same object across views and applies LeJEPA SIGReg to prevent collapse. Its instance space uses a contrastive loss to separate an object's patches from context and other objects. Copy-paste augmentation places an object into a new background to reduce context leakage.

The paper reports ADE20k object-classification balanced accuracy of 0.555 for its semantic object representation and 0.426 for image-level LeJEPA. It reports DAVIS tracking J&F of 0.683 and 0.626, respectively. The paper says each configuration has one training seed. Its COCO size study uses 15,000 steps for subsets up to 1,024 images, and 69,000 steps for 10% and full COCO. The main setup uses ViT-B/16, batch size 256 across four RTX A6000 GPUs, and a 300-epoch COCO schedule.

The paper's reproducibility section says training and SAM-mask code will be released publicly. The current v2 paper does not link an object-centric code repository. The separate [official image-level LeJEPA repository](https://github.com/galilai-group/lejepa) provides the original SIGReg implementation and baseline method, but it is not the object-centric paper's implementation.

## Local study choice

Use [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) through the [TorchVision dataset interface](https://docs.pytorch.org/vision/0.28/generated/torchvision.datasets.OxfordIIITPet.html). The Oxford source lists 37 pet categories and a pixel-level foreground/background trimap for each image. It lists a total download size of about 800 MB and a CC BY-SA 4.0 license; image copyright remains with the original owners.

Compare the image-pooled and foreground-mask-pooled LeJEPA losses using the repository's Epps–Pulley SIGReg helper, a small CNN, a balanced trainval subset, and a frozen breed-classification probe on the official test split. Use foreground plus outline trimap values as the foreground region. Keep labels out of the self-supervised loss.

The checksum-verified official archives loaded on 2026-10-14 contain seven trainval and seven test trimaps with only background pixels. The local protocol excludes empty trainval masks before balanced sampling. It retains every official test example and uses the same global-average pooling fallback for both methods when a test trimap has no foreground.

This is an exploratory transfer study, not a reproduction of the paper's result. The dataset, backbone, input size, training budget, evaluation task, and parts of the objective differ. Oxford-IIIT Pet has one main pet per image, so the study omits and cannot validate the paper's within-image instance-separation loss or its copy-paste augmentation. The exact COCO benchmark remains open.

## Decision and gates

- Do not claim the local result confirms or refutes the paper's COCO findings.
- Keep paper-reported and locally measured values in separate tables.
- Do not add object-centric code from an unofficial repository as though it were the authors' implementation.
- Revisit an exact COCO result when first-party code, a bounded resource plan, the paper's selected dataset, and the matching evaluation are available.

## Sources

- Geusen and Konukoglu, [arXiv v2 paper and HTML](https://arxiv.org/html/2607.02404), 2026-10-05.
- Authors' [official image-level LeJEPA code](https://github.com/galilai-group/lejepa).
- Parkhi et al., [Oxford-IIIT Pet dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/), 2012.
- [TorchVision Oxford-IIIT Pet dataset documentation](https://docs.pytorch.org/vision/0.28/generated/torchvision.datasets.OxfordIIITPet.html).
