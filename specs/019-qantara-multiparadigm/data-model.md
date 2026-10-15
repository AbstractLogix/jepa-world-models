# Data Model: Qantara Multi-Paradigm Evaluation

## Entities

### Paper target

- `paper_version`: arXiv v2 dated 2026-10-10.
- `table`: Table 2, Push-T row.
- `metric`: episode success rate.
- `reported_values`: CEM 90.1 ± 1.1; BC 82.1 ± 1.0; video–inverse 83.2 ± 0.6.
- `aggregation`: 50 episodes × five evaluation seeds × three training seeds; population standard deviation across training-seed means.

### Checkpoint

- `method`: Qantara.
- `training_seed`: 11, 22, or 33.
- `hub_revision`: `de5f7575c9db4a564382e4b2b681fa534dcb3b62`.
- `filename`, `size_bytes`, `hub_sha256`, `local_sha256`, and `load_status`.
- Serialized object loads only in a controlled environment after source and artifact review.

### Dataset

- `source`: DINO-WM original Push-T archive or a verified conversion.
- `revision`, `filename`, `size_bytes`, `hub_sha256`, and `local_sha256`.
- `episode_manifest`: episode IDs, lengths, source identity, and conversion record.
- `license`: source card and applicable upstream terms.
- The official Qantara config expects HDF5 named `pusht_expert_train`; data identity must be verified before use.

### Evaluation run

- `training_seed`, `inference_path`, `evaluation_seed`, `episode_id`, `start_step`, and `goal_step`.
- `success`, `steps`, `elapsed_seconds`, `peak_memory_bytes`, and `status`.
- `code_revision`, `runtime_versions`, and `deviations`.
- Raw records are stored outside Git and referenced by path and hash in the report.

## Validation rules

- Every successful evaluation case must map to one checkpoint, one inference path, one declared evaluation seed, and a source episode/step pair.
- The three inference paths for a given checkpoint and evaluation seed must use the same case set.
- Missing or failed episodes stay in the run record and prevent a complete status.
- Paper values and local measurements are separate fields.
- A run with a different local seed set is not an exact sample-level reproduction.
