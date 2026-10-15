# Quickstart: LeHami Actuated-Pendulum Velocity Reproduction

This procedure uses an isolated official-code clone. It does not change the project dependencies. Keep generated data and checkpoints under `/tmp`.

## 1. Check local hardware and official sources

Use a CUDA-capable NVIDIA host. Confirm the LeHami clone is at commit `f76f65f8b99afb6b7e3063f573645d3e2f290ede` and clone the LeWorldModel dependency at commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. Do not clone DINO-WM or download DINOv2 weights; this comparison does not use them.

## 2. Install the isolated training environment and check CUDA

The official `env` extra includes unrelated simulator packages. On this host, `labmaze` in that extra needs Bazel. The selected pendulum scripts do not use those packages. Create a temporary project from the official manifest, remove only that extra, and install its remaining locked dependencies:

```bash
mkdir -p /tmp/lehami-runtime-env
cp pyproject.toml uv.lock /tmp/lehami-runtime-env/
cd /tmp/lehami-runtime-env
sed -i 's/stable-worldmodel\[train,env\]/stable-worldmodel[train]/' pyproject.toml
uv lock --python python3.13
uv sync --locked --no-dev --no-install-project --python python3.13
```

Create `/tmp/lehami-uv-python` as a small wrapper that removes the `run python` arguments and calls `/tmp/lehami-runtime-env/.venv/bin/python`. Set `UV=/tmp/lehami-uv-python` when calling the official shell scripts. Confirm `torch.cuda.is_available()` before training. Keep the official LeHami source and lockfile unchanged. Use a separate `STABLEWM_HOME` for each model arm so the runner skips the unrelated Jacobian-volume evaluation.

## 3. Generate and verify the dataset

Generate the data once and copy the verified 15.4 MB HDF5 file into each separate model home:

```bash
UV=/tmp/lehami-uv-python STABLEWM_HOME=/tmp/lehami-a0q-home bash lehami/data/prepare_actuated_data.sh
cp /tmp/lehami-a0q-home/pendulum_swingup.h5 /tmp/lehami-b0s-home/pendulum_swingup.h5
```

Compare each `pendulum_swingup.h5` hash with `dataset_sha256` in `lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json`. Stop if either hash differs.

## 4. Run an isolated one-epoch smoke check

Use a third data home for each model and override `trainer.max_epochs=1` for seed 0. Train with the official `lehami/training/train.py` entry point. Then run `scripts/reproductions/lehami_k0_probe.py` with the matching checkpoint, manifest, and `STABLEWM_HOME`. It calls the official K=0 feature collector and 25-NN helper, verifies the data hash, and writes `metrics.json`, `probe_knn.pkl`, and `test_predictions.npz`. The full experiment runner also fits unrelated MLP and K=1 probes; those are outside this smoke check. Save smoke outputs separately. Do not count smoke metrics as paper evidence.

## 5. Run the selected three-seed comparison

For each of `a0q` and `b0s`, run the official `lehami/training/train.py` entry point with its config, manifest, and seeds `0`, `1`, and `2`; do not override training settings. The configs set 35 epochs and batch size 32. The first user-selected pass runs seed 0 for each model; this is partial evidence. Seeds 1 and 2 are required to complete the paper's three-seed result.

For each finished checkpoint, run `scripts/reproductions/lehami_k0_probe.py` with the same official source checkout, manifest, and model `STABLEWM_HOME`. Save its JSON metric, fitted probe, and held-out predictions with that run's outputs. The script checks that the data hash matches the shipped manifest. Do not use frozen values from `paper_figures/data/` as local results.

## 6. Record evidence and align the roadmap

Write paper values and local seed values side by side in `docs/reproductions/lehami-actuated-velocity.md`. Include environment, data hash, manifest, runtime, deviations, and limitations. Update the roadmap, reproduction matrix, coverage map, and project task list only after the evaluation finishes. If any required run or integrity check is missing, mark the result partial.
