# Data Model: Object-centric LeJEPA Lesson

## Data sources

| Source | Use | Split | Local storage |
| --- | --- | --- | --- |
| Oxford-IIIT Pet images | Unlabeled pretraining views and frozen features | `trainval`, fixed balanced subset; `test`, full | `notebooks/data/oxford-iiit-pet` |
| Oxford-IIIT Pet trimaps | Mask pooling for object-level training and both-arm object probes | Same image split | Same dataset cache |
| Breed categories | Linear-probe labels only | `trainval` for fit, `test` for final score | Loaded from dataset metadata |

The trimap identifies pet, background, and boundary pixels. The notebook treats pet and boundary pixels as foreground. Resize masks with nearest-neighbour interpolation. Exclude trainval examples with no foreground at the input size before balanced sampling. Retain the full test split; use the same global-average pooling fallback for both methods when a test trimap has no foreground.

## Run record

Each row contains:

- method: `image_level` or `mask_pooled`
- seed and fixed subset fingerprint
- method configuration and number of completed epochs
- trainval probe size and test size
- balanced accuracy and linear-probe status
- training and feature-extraction duration
- Python, PyTorch, TorchVision, NumPy, SciPy, scikit-learn versions
- device and available peak CUDA memory
- count of trainval examples excluded for empty masks and test examples using the shared global-pooling fallback
- error text and completion state

Failed seeds stay in the run table. Aggregates use completed runs only and state the completion count.

## Summary record

For each method, store completed seed values, mean, median, sample standard deviation, and a 95% Student-t interval over independent seeds. For each matched seed, also store the object-minus-image balanced-accuracy difference and its mean, median, and interval. The paper's reported results stay in a separate citation-backed record.
