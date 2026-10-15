# Data Model: V-JEPA 2.1 Dense Video Features Tutorial

This notebook has no persistent database. The following records describe one local execution.

## Video clip

| Field | Type | Rule |
|---|---|---|
| frames | RGB tensor/list | Exactly 16 deterministic frames; 3 color channels. |
| height, width | integers | Both 384; divisible by patch size 16. |
| object_centers | 16 coordinate pairs | Deterministic and inside the frame. |
| generation_seed | integer | Fixed and printed so changes are visible. |

## Checkpoint record

| Field | Type | Rule |
|---|---|---|
| source_repository | URL | Official `facebookresearch/vjepa2`. |
| code_revision | Git SHA | Fixed to `204698b45b3712590f06245fbfba32d3be539812`. |
| model_entrypoint | string | `vjepa2_1_vit_base_384`. |
| checkpoint_url | URL | Official `vjepa2_1_vitb_dist_vitG_384.pt`. |
| checkpoint_bytes | integer | Expected source observation: 1,664,223,428 bytes; report actual downloaded file size. |
| terms_status | string | Checkpoint terms are not separately specified in the repository README; learner must check current source terms. |
| cache_path | local path | Under ignored `notebooks/checkpoints/`; never committed. |

## Feature grid

| Field | Type | Rule |
|---|---|---|
| tensor | float tensor | Finite values with batch, time, patch, and channel axes identified. |
| temporal_groups | integer | 8 for 16 frames and tubelet size 2. |
| patch_rows, patch_columns | integers | 24×24 for 384-pixel input and 16-pixel patches. |
| device | string | Actual selected PyTorch device. |
| preprocessing | string | Official transform name/revision and resulting input shape. |
| elapsed_seconds | float | Measured locally around model inference. |
| peak_memory_bytes | integer or unavailable | Report process peak when measured; otherwise state unavailable. |

## Correspondence result

| Field | Type | Rule |
|---|---|---|
| query_time_group | integer | Within 0–7. |
| known_target_patch | row/column pair | Derived from the generated object center. |
| retrieved_patch | row/column pair | Highest cosine similarity at the later time group. |
| cosine_similarity | float | Finite value for the retrieved patch. |
| target_distance_patches | float | Euclidean distance from retrieval to known target in patch coordinates. |
| baseline | string | Mean-RGB patch descriptor with the same query and target. |
| interpretation | string | Descriptive measurement only; no pass threshold or general claim. |

## Relationships

One generated clip produces one feature grid when the checkpoint is loaded. One correspondence record is produced per named feature method, with matching clip, query, target, preprocessing, and run conditions.
