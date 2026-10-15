# Data Model: VL-JEPA Real Label Embeddings

## CIFAR-10 sample

- **Fields**: image, integer class ID, split (`train` or `test`), source index.
- **Rules**: Select exactly 100 images per class from each official split using the fixed data seed `20261014`. Never move an item between the official splits.
- **Derived value**: Frozen, 512-dimensional ResNet-18 feature using the official ImageNet preprocessing.

## Class prompt

- **Fields**: CIFAR-10 class name, fixed prompt, 384-dimensional text embedding.
- **Rules**: Create one prompt per class with a fixed template. Encode with `sentence-transformers/all-MiniLM-L6-v2` at revision `ea78891063587eb050ed4166b20062eaf978037c`; apply attention-mask mean pooling and L2 normalization.
- **Derived value**: Candidate prompt similarity for a predicted normalized embedding.

## Method run

- **Fields**: method name, initialization seed, train/test features, model settings, test predictions, per-class correct/total counts, top-1 accuracy.
- **Rules**: Methods use the same sampled images and cached visual features. The embedding predictor uses cosine loss and ranks the ten class prompt embeddings. The linear baseline uses ten output logits and cross-entropy. Seeds are fixed to `[7, 17, 27, 37, 47]`.
- **Invalid state**: Non-finite loss, embedding, or score; incomplete class sample; missing test prediction.

## Result summary

- **Fields**: software versions, hardware, data seed, method settings, per-seed accuracies, mean, median, standard deviation, uncertain 95% interval across seeds, paired per-image accuracy difference and class-stratified bootstrap interval, class-wise accuracy, raw predictions.
- **Rules**: Report both methods separately. Label five-seed intervals uncertain. Keep the per-image interval distinct from training-seed variation. Do not combine the paper's values with local values. State that this is a closed-set, label-supervised analogue.
