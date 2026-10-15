# Data model: Paper-aligned JEPA objectives

## CIFAR-10 split

| Field | Value |
|---|---|
| Dataset | Official CIFAR-10, 50,000 train and 10,000 test examples |
| Self-supervised subset | 640 balanced train images (64 per class); images only reach the encoder |
| Probe train subset | 1,000 different balanced train images; 100 labels per class |
| Final test subset | 1,000 balanced test images; 100 labels per class |
| Selection seed | `20261014` |
| Preprocessing | 32×32 RGB input; fixed CIFAR-10 normalization; notebook 02 uses two global and four local crops; notebook 03 uses two stochastic crops |
| Data location | `notebooks/data`; never add to Git |

The self-supervised images, probe-training images, and final-test images have disjoint dataset indices. The method receives no class labels during self-supervised training. Labels are used only to balance the selected subsets and to fit/evaluate the linear probes.

## Model and run

- Backbone: four convolution stages with channels 32, 64, 128, and 256; global pooling; 256-dimensional output. No pretrained weights.
- Rectified projector: three linear layers with width 256, batch normalization and ReLU in the two hidden layers, and a final ReLU for paper-aligned RDMReg.
- Run key: `(notebook, objective, training_seed)`.
- Training seed: one of `[7, 17, 27, 37, 47]`; controls initialization, data order, augmentations, projections, and target samples.
- Training budget: 20 epochs, batch size 128, AdamW learning rate `0.003`, weight decay `1e-5`. The frozen logistic-regression probe uses `max_iter=1000`.
- Views: notebook 02 uses two global and four local views in both arms; notebook 03 uses two views in both arms. The paper recommends two global and eight local views for LeJEPA.
- Projection count: 128 for the current LeJEPA teaching arm; 512 for the paper-aligned SIGReg and RDMReg arms.

## Metrics

- **Top-1 linear-probe accuracy**: fraction of the fixed CIFAR-10 test subset classified correctly by a logistic regression model trained on the fixed probe-training subset and frozen features.
- **Embedding standard deviation**: mean standard deviation across feature coordinates for the LeJEPA representation; values near zero can indicate collapse.
- **Mean absolute off-diagonal covariance**: mean absolute value of non-diagonal entries of the LeJEPA feature covariance matrix; it measures residual coordinate correlation.
- **L0 sparsity**: fraction of nonzero coordinates per Rectified LpJEPA projector output; lower means more sparse.
- **L1 sparsity**: `(1/D) * (||z||_1 / ||z||_2)^2` per projector output; lower means more sparse. Define the zero-vector result as zero and report it beside linear-probe accuracy.
- **Run status**: `completed`, `failed`, or `interrupted`, with error and elapsed time where applicable.

Aggregate each metric across the five training seeds. Report every seed, mean, median, and a two-sided 95% Student-t interval. Treat that interval as uncertain due to the small run count.
