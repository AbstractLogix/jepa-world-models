# Research: VL-JEPA Real Label Embeddings

**Reviewed**: 2026-10-14

## Paper findings

Primary source: [VL-JEPA, arXiv:2512.10942, version 2](https://arxiv.org/html/2512.10942), dated 2026-02-02.

- The paper defines an X-Encoder for visual embeddings, a Predictor for target-text embeddings, a Y-Encoder for real text targets, and a Y-Decoder that is used only when text output is needed.
- The reported implementation uses frozen V-JEPA 2 ViT-L (304M parameters), eight Llama-3.2-1B layers in the predictor (about 490M parameters), and EmbeddingGemma-300M. The shared projected space has 1,536 dimensions.
- The paper trains on DataComp, YFCC-100M, and Action100M. Its reported pretraining includes 2B samples and four weeks on 24 nodes with eight H200 GPUs per node. Its base model is evaluated on eight video classification and eight text-to-video retrieval datasets.
- The paper reports 1.6B parameters for the full system and claims about 50% fewer trainable parameters than its controlled token-prediction comparison and about 2.85x fewer decoding operations for selective decoding. These are paper claims, not results from this repository.
- The paper text reviewed here does not link an implementation repository or a released VL-JEPA checkpoint. A first-party code/checkpoint source was not confirmed in this review. This remains a blocker for an exact reproduction.

## Local data and model sources

| Item | Decision and source | Terms and size |
|---|---|---|
| Image data | CIFAR-10 Python archive from the [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html). Use 100 images per class from the official train split and 100 per class from the official test split. | The Python archive is 163 MB and the page asks users to cite the CIFAR technical report. The page reviewed does not state dataset license terms. Do not redistribute it; learners must check terms for their use. |
| Image weights | `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`; official [TorchVision model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) and [weight source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py). | 44.7 MB. TorchVision warns that pretrained models can have separate terms derived from training data. The cited sources do not confirm a specific license for these weights. |
| Text weights | `sentence-transformers/all-MiniLM-L6-v2`, revision `ea78891063587eb050ed4166b20062eaf978037c`; [model card at that revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/ea78891063587eb050ed4166b20062eaf978037c). | Apache-2.0 in the model card. The model is 384-dimensional; its `model.safetensors` file is 90.9 MB. Use attention-mask mean pooling and L2 normalization as described by the model card. |
| Runtime | Optional Transformers 5.x and Safetensors; PyTorch remains installed separately under the project instructions. | Transformers 5.19.0 is the current version found on PyPI during this review and supports Python 3.10+. Use a compatible range `>=5,<6`; verify the exact installed version at run time. |

## Decisions

### Use class labels, not captions

The official CIFAR-10 source provides class labels but no paired captions. Encode one natural prompt per class (for example, “a photo of a dog”). This uses real language embeddings but does not test image-caption alignment or caption retrieval.

### Compare two methods on the same visual features

Freeze the same ResNet-18 features for both methods. The embedding method learns a small predictor from the 512-dimensional visual features to normalized MiniLM class-prompt embeddings with cosine loss. The baseline is a linear ten-class classifier with cross-entropy. Both use identical fixed train/test examples and five initialization seeds. Report classification top-1 and class-wise results. This is a practical baseline comparison, not a capacity-matched study.

### Pin source assets and keep limits explicit

Pin the MiniLM model to an immutable Hugging Face revision. Use the named TorchVision weight enum and the installed TorchVision version. State data and weight sizes before download. Because the CIFAR-10 and ImageNet training-data terms remain unclear in the reviewed sources, the project will not redistribute the data or weights.

### Do not claim paper reproduction

The local task lacks the paper's V-JEPA 2 encoder, Llama predictor, EmbeddingGemma target encoder, video-caption training mixture, scale, decoder, benchmarks, and official training recipe. It demonstrates one mechanism only. Exact reproduction work stays open until the authors' code, weights, data access, and benchmark setup are independently verified.

## Alternatives considered

- **Use Notebook 04 alone**: Rejected because it should remain a small synthetic introduction; replacing its exercise would remove that progression and would not supply a separate advanced lesson.
- **Use generated captions**: Rejected because generated captions would add another model and unverified target quality. The official dataset already provides real labels.
- **Use the VL-JEPA paper's model**: Deferred because this review did not confirm a first-party implementation/checkpoint, and the paper's training scale is outside this local tutorial's resource scope.
- **Add `sentence-transformers`**: Rejected because the model card documents how to create embeddings with Transformers directly. This keeps the optional dependency smaller and exposes the pooling step for teaching.

## Sources

- [VL-JEPA paper, arXiv v2](https://arxiv.org/html/2512.10942)
- [CIFAR-10 official page](https://www.cs.toronto.edu/~kriz/cifar.html)
- [MiniLM model card at pinned revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/ea78891063587eb050ed4166b20062eaf978037c)
- [TorchVision ResNet-18 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [TorchVision ResNet model source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [TorchVision model and dataset terms notice](https://github.com/pytorch/vision/blob/main/README.md#disclaimer-on-datasets)
- [Transformers project release history](https://pypi.org/project/transformers/)
