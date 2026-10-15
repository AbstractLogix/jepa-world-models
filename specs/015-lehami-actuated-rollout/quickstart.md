# Quickstart: LeHami Seed-0 Actuated Rollout Evaluation

## Prerequisites

- The pinned official LeHami checkout and its existing Python 3.13 environment.
- The full seed-0 `a0q` and `b0s` checkpoints.
- `pendulum_swingup.h5` whose hash matches the official manifest.
- CUDA, as used in the recorded local run.

No model training is required.

## Run

The official runner writes to `lehami/experiments/results/<exp-name>/`. Use a new name for each run. From the repository root, run the following command once for each model, with the matching checkpoint path:

```sh
STABLEWM_HOME=/path/to/seed0-data-home \
  /path/to/lehami-python \
  lehami/experiments/src/run_eval.py \
  --checkpoint /path/to/best.ckpt \
  --manifest lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json \
  --exp-name unique-seed0-rollout-name \
  --horizon 76 --device cuda --knn-neighbors 25 --rollout-seed 0
```

Use the standard mode. Do not pass `--zero-action` or `--skip-extra-probes`. Standard mode fits probes on the training split and applies the K=1 probe frozen during rollout evaluation.

## Expected Results

- Each model has raw `probe_metrics.csv`, `rollout_metrics.csv`, and `summary.json` output in its unique result directory.
- Test-split angular-velocity MAE is available by rollout step for the K=1-fitted 25-neighbor probe.
- The local result has one training seed per model. It cannot reproduce the paper's three-seed mean or standard deviation.
- The evaluator measures prediction only. It does not measure planning success or physical-system validity.
