# VL-JEPA-Inspired CIFAR-10 Label-Embedding Analogue

**Claim status**: Locally measured illustrative analogue. This is not a VL-JEPA reproduction, zero-shot result, or open-vocabulary evaluation.

**Notebook**: [`notebooks/11_vl_jepa.ipynb`](../../notebooks/11_vl_jepa.ipynb)

## Research context

The [VL-JEPA paper, arXiv version 2](https://arxiv.org/html/2512.10942) describes an image encoder, a predictor of text embeddings, a text encoder, and a decoder used when text output is needed. The paper reports a 1.6B-parameter system and video classification, retrieval, and question-answering results using V-JEPA 2 ViT-L, a Llama-3.2-based predictor, EmbeddingGemma, and large-scale image/video data.

The local lesson tests only the idea of predicting real target-text embeddings. It uses class prompts, not captions, and does not use the paper's model stack, data, training recipe, decoder, or benchmarks. This run does not validate the paper's findings. A first-party VL-JEPA implementation and checkpoint were not confirmed in the paper source review.

## Data and model provenance

| Component | Local run | Source and limits |
|---|---|---|
| Data | CIFAR-10, 100 samples per class from each official split; 1,000 train and 1,000 held-out images. The files were already present before this run. | [Official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html) describes 50,000 train images, 10,000 test images, and a 163 MB Python archive. It asks users to cite the technical report but does not state license terms. TorchVision accepted the existing files through its dataset integrity checks; their original acquisition date is unknown. |
| Image encoder | Frozen `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`; 512-dimensional features. | [TorchVision documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) lists 44.7 MB and the official resize/crop/normalization. The checkpoint URL uses `resnet18-f37072fd.pth`. TorchVision warns that model weights can have separate terms; a specific weight license was not confirmed in the sources reviewed. |
| Text encoder | Frozen `sentence-transformers/all-MiniLM-L6-v2`, revision `ea78891063587eb050ed4166b20062eaf978037c`; 384-dimensional mean-pooled, normalized outputs. | [Model card at the pinned revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/ea78891063587eb050ed4166b20062eaf978037c) lists Apache-2.0 and a 90.9 MB model file. |

The notebook declares the official CIFAR-10 source and downloads the named checkpoints when absent. This run used the pre-existing CIFAR files and downloaded the ResNet weights. The assets and generated outputs remain outside version control.

## Run conditions

| Setting | Value |
|---|---|
| Platform | Linux WSL2, kernel `6.18.54.1-microsoft-standard-WSL2`, x86_64 |
| Python | 3.13.14 |
| PyTorch / TorchVision | 2.10.0+cpu / 0.25.0+cpu |
| Transformers / Safetensors | 5.19.0 / 0.8.0 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| Device | CPU; no CUDA device available |
| Data seed | `20261014` |
| Training seeds | `7, 17, 27, 37, 47` for each method |
| Training subset | 100 images per class; 1,000 total |
| Held-out subset | 100 images per class from the official test split; 1,000 total |
| Head training | 25 epochs, batch 128, AdamW, learning rate `0.001`, weight decay `1e-4` |
| Feature extraction | Batch 64; 37.8 seconds for 2,000 images |
| Ten head runs and inference | 3.5 seconds |
| Peak memory | Not measured |

Both methods use the same cached frozen visual features and the same data subset. The text predictor uses a two-layer 512→256→384 MLP and cosine loss. The baseline is a 512→10 linear classifier trained with cross-entropy. The methods have different output sizes and objectives, so this is not a capacity-matched comparison.

## Local measurements

Top-1 accuracy on the fixed 1,000-image test subset:

| Method | Seed 7 | Seed 17 | Seed 27 | Seed 37 | Seed 47 | Mean | Median | SD | 95% t interval across seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text-embedding predictor | 0.808 | 0.805 | 0.794 | 0.805 | 0.802 | 0.803 | 0.805 | 0.005 | [0.796, 0.809] |
| Linear classifier | 0.792 | 0.798 | 0.811 | 0.800 | 0.802 | 0.801 | 0.800 | 0.007 | [0.792, 0.809] |

The intervals use a Student t interval with four degrees of freedom. Five runs give an uncertain estimate of training-seed variation. The methods are close in this run.

The paired per-image accuracy difference, averaged across the five runs, was **+0.002** for the text predictor. Its 95% class-stratified bootstrap interval on this fixed test sample was **[-0.010, 0.015]** from 5,000 resamples. The interval includes zero and does not show a clear advantage. It describes test-image variation for these five trained models, not training-seed variation.

Mean class-wise top-1 accuracy across the five runs:

| Class | Text predictor | Linear baseline |
|---|---:|---:|
| airplane | 0.820 | 0.792 |
| automobile | 0.910 | 0.908 |
| bird | 0.732 | 0.682 |
| cat | 0.650 | 0.656 |
| deer | 0.794 | 0.844 |
| dog | 0.766 | 0.736 |
| frog | 0.846 | 0.856 |
| horse | 0.762 | 0.778 |
| ship | 0.858 | 0.836 |
| truck | 0.890 | 0.918 |

All ten predictor runs and all ten baseline runs completed. Each used all 1,000 held-out images; no run was missing or dropped.

## Interpretation and limits

The local experiment shows that a small predictor can map frozen image features to real class-prompt embeddings and rank those prompts. Its mean accuracy was 0.2 percentage points above the linear baseline, but the paired interval includes zero. The result does not support a claim that the embedding objective is better.

The sample is small and restricted to ten classes. Both methods receive class supervision, and the image encoder was pretrained on ImageNet. The text model is a general sentence embedder, not the paper's text encoder. The run uses no captions, video, VL-JEPA checkpoint, language decoder, or paper benchmark. It is a local teaching comparison only.
