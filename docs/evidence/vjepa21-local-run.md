# V-JEPA 2.1 Local Run

**Run date**: First run 2026-10-14; fresh-kernel repeat 2026-10-15  
**Status**: Local checkpoint execution repeated; this is not a paper reproduction.

## Setup and input

- Notebook: `notebooks/09_vjepa21_dense_features.ipynb`
- Model: `vjepa2_1_vit_base_384`
- Official code revision: [`204698b45b3712590f06245fbfba32d3be539812`](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)
- Checkpoint: [`vjepa2_1_vitb_dist_vitG_384.pt`](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt)
- Checkpoint size: 1,664,223,428 bytes (about 1.66 GB)
- Checkpoint SHA-256: `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`
- Terms: the official README does not specify separate checkpoint terms. Check current terms before use or redistribution.
- Input: one deterministic synthetic clip, seed 21, 16 RGB frames at 384×384 pixels. No external video data was used.
- Official preprocessing: pinned `vjepa2_preprocessor(crop_size=384)` evaluation path with resize, center crop, tensor conversion, and ImageNet normalization.
- Preprocessed model input: `[1, 3, 16, 384, 384]`.

## Machine and run

- Device: CPU; CUDA was unavailable.
- CPU: AMD Ryzen 9 5900X; PyTorch used 8 threads.
- Python: 3.12.13
- PyTorch: 2.10.0+cpu
- Torchvision: 0.25.0+cpu
- Available system memory before model loading: 27.73 GiB.
- Available system memory after inference: 26.68 GiB.
- Process lifetime peak resident memory after inference: 2.28 GiB, as reported by `resource.getrusage`.
- Cached-checkpoint model load: 4.632 seconds.
- Encoder inference: 3.505 seconds.
- The first run downloaded the 1.66 GB checkpoint and completed the load path in about 150 seconds. Later runs use the local checkpoint cache.

## Observed output

- Feature shape: `[1, 8, 24, 24, 768]` (batch, temporal groups, patch rows, patch columns, channels).
- All feature values passed the finite-value check.
- Query group: 1. Target group: 6. Known target patch: row 10, column 17.
- V-JEPA cosine retrieval: row 11, column 17; cosine similarity 0.9531; distance from the known patch 1 grid cell.
- Mean-RGB baseline: row 11, column 17; cosine similarity 0.9990; distance from the known patch 1 grid cell.

Both retrievals were one patch from the known location on this clip. There is no pass threshold. This single synthetic measurement does not establish general correspondence quality or reproduce any V-JEPA 2.1 benchmark.

The notebook's no-download path also completed. It generated the same 16×384×384 clip and clearly skipped model loading and correspondence.

## Fresh-kernel repeat (2026-10-15)

The opt-in path ran again from a fresh Jupyter kernel with the existing checkpoint and pinned source cache. It did not transfer another checkpoint or video. The resulting notebook is `/tmp/vjepa21-current-run-20261015.ipynb`.

- Python 3.12.13; PyTorch 2.10.0+cpu; Torchvision 0.25.0+cpu; CPU with 8 threads.
- The checkpoint had the expected 1,664,223,428 bytes and SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.
- Model load took 12.379 seconds; encoder inference took 3.200 seconds. Peak resident memory was 2.27 GiB.
- The notebook returned finite features with shape `[1, 8, 24, 24, 768]`.
- Both the V-JEPA and mean-RGB retrievals selected row 11, column 17, one grid cell from the known patch. Cosine similarities were 0.9531 and 0.9990.
- The saved notebook passed `nbformat` validation and had no cell errors.

This repeat confirms local execution for the named checkpoint and synthetic clip. It does not measure general feature quality or reproduce a paper benchmark.

## Re-run

Install the project dependencies, the `vjepa21` optional extra, and a PyTorch wheel that matches the machine. Then run this command from the repository root:

```bash
JEPA_LOAD_VJEPA21=1 JEPA_CPU_THREADS=8 jupyter nbconvert \
  --to notebook --execute notebooks/09_vjepa21_dense_features.ipynb \
  --output vjepa21-run.ipynb --output-dir /tmp
```

The model load downloads the checkpoint when it is not in the local cache. The notebook keeps model loading opt-in. The downloaded checkpoint is not part of the repository.
