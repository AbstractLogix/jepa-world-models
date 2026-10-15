# Research: V-JEPA 2.1 Dense Video Features Tutorial

**Research date**: 2026-10-14

## Official model and checkpoint

- **Decision**: Use the smallest checkpoint in the official V-JEPA 2.1 repository: `vjepa2_1_vit_base_384`, ViT-B/16, 80M parameters. The official README lists its resolution as 384 and links to `vjepa2_1_vitb_dist_vitG_384.pt`.
- **Rationale**: It is the smallest released V-JEPA 2.1 model and supports a feasible teaching example. The lesson needs a real released encoder, while keeping model weights optional.
- **Alternatives considered**: The larger ViT-L and ViT-G checkpoints require more resources. A randomly initialized encoder does not teach released-model feature inspection.
- **Primary sources**: [Official V-JEPA 2 repository and checkpoint table](https://github.com/facebookresearch/vjepa2); [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482).
- **Checkpoint size**: The checkpoint URL returned HTTP 200 and a `Content-Length` of 1,664,223,428 bytes during research. This is a source observation, not a promise that the server will retain the same file.
- **Terms**: The repository README describes code licensing, but does not separately specify checkpoint terms. The notebook must show this as “not separately specified in the official README,” link the source, and ask learners to check current terms. Do not redistribute the checkpoint.

## Pinned loading path

- **Decision**: Pin code to Git commit `204698b45b3712590f06245fbfba32d3be539812`; call the Hub entrypoint with `pretrained=False`, then fetch the official checkpoint only after explicit opt-in and strictly load its `ema_encoder` weights.
- **Rationale**: At the pinned revision, the official source defines its default checkpoint base URL as `http://localhost:8300`; using `pretrained=True` would depend on a local server. Manual loading from the published checkpoint URL avoids this hidden failure and makes the download visible.
- **Alternatives considered**: Floating `main` is not reproducible. `pretrained=True` is concise but fails at this revision without a localhost service. Reimplementing the model loader would duplicate upstream code and increase drift risk.
- **Primary sources**: [Pinned upstream source: Hub backbone loader](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/hub/backbones.py); [pinned upstream configuration](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train_2_1/vitb16/pretrain-256px-16f.yaml); [official checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt).

## Input size and temporal layout

- **Decision**: Generate 16 RGB frames at 384×384. Use 16-pixel patches and 2-frame tubelets, yielding eight temporal groups and a 24×24 spatial patch grid.
- **Rationale**: The public `vjepa2_1_vit_base_384` entrypoint fixes `img_size=384`. It accepts an override for `num_frames`, so the lesson uses 16 frames to reduce sequence length while retaining the published image size. The official preprocessor accepts `crop_size=384` and returns a C×T×H×W tensor after resize, center crop, conversion, and normalization.
- **Alternatives considered**: Passing `img_size=256` through the public wrapper is invalid because the wrapper supplies its own `img_size=384`; using the private `_make_vjepa2_1_model` factory would rely on an internal API. Four frames produce only two temporal groups and are a weaker correspondence demonstration.
- **Primary sources**: [Pinned Hub model entrypoint](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/hub/backbones.py); [pinned Hub preprocessor](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/evals/hub/preprocessor.py); [pinned video transforms](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/evals/video_classification_frozen/utils.py).

## Dependencies and preprocessing

- **Decision**: Keep PyTorch installation separate, use existing `timm`, and add a `vjepa21` optional extra for `einops` and `opencv-python-headless`. Use the official Hub preprocessing functions and report the actual preprocessing path in each model run.
- **Rationale**: The official Hub entrypoint declares `torch`, `timm`, and `einops`; the official Hub preprocessor imports video transforms that require OpenCV. Core clip generation should not require these model dependencies or network access.
- **Alternatives considered**: Adding video decoding libraries is unnecessary for generated RGB frames. An undocumented transitive dependency on `einops` would make the opt-in path brittle.
- **Primary source**: [Official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2), including its pinned Hub implementation and evaluation preprocessing code.

## Evaluation boundary

- **Decision**: Measure one synthetic moving-object patch retrieval and compare it with mean-RGB patch features using cosine similarity. Report results, not a success threshold.
- **Rationale**: The synthetic path is deterministic and cheap. It helps learners inspect what features encode without implying that it validates real-video tracking or reproduces paper benchmarks.
- **Alternatives considered**: A real-video benchmark requires licensed data and a defined annotation set; a single synthetic clip is not evidence for benchmark-level performance.
- **Primary sources**: [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482); [RLiable evaluation methodology](https://arxiv.org/abs/2108.13264).
