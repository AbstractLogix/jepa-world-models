# LeHami Actuated-Pendulum Velocity Probe

**Status**: Partial reproduction; seed-0 pair complete, seeds 1 and 2 pending  
**Reviewed**: 2026-10-15  
**Feature**: [013](../../specs/013-lehami-actuated-reproduction/spec.md)

## Claim under study

LeHami Table 1 reports held-out angular-velocity MAE at K=0 with a 25-nearest-neighbor probe. Its reported mean ± standard deviation over three seeds is 0.419 ± 0.045 rad/s for `a0q`; the LeWM `b0s` baseline is 2.508 ± 0.002 rad/s. These are paper values, not local results.

The report also contains a separate K=1-fitted, frozen-probe rollout check for seed 0. The K=0 Table 1 probe and the multi-step rollout probe use different measurements and must stay separate. Neither probe tests planning, control, or physical validity. The paper's exact symplecticity claim is a structural property; it does not bound rollout error.

## Selected local scope

This local comparison follows the [project evaluation protocol](../EVALUATION_PROTOCOL.md). Both models completed 35 epochs with seed 0 and their official batch-32 configs. Seeds 1 and 2 remain pending. The one-seed-per-model result is a partial reproduction attempt; it cannot reproduce the paper's three-seed mean or standard deviation.

| Model | Official config | Seed | Full run status | Training wall time | K=0 25-NN MAE |
| --- | --- | ---: | --- | --- | --- |
| LeWM baseline | `phase0_lewm_ms_conv_w1_swingup_exp072_b0s.yaml` | 0 | Completed, 35 epochs | 13,560.75 s (3 h 46 min 1 s) | 2.504686 rad/s |
| LeHami | `phase2_lehami_sg_sep_conv_w1_swingup_exp079_a0q.yaml` | 0 | Completed, 35 epochs | 8,829.73 s (2 h 27 min 10 s) | 0.456258 rad/s |

Both probe runs used CUDA, the official 25-NN helper, the verified dataset hash, 630 training episodes (49,770 rows), and 135 held-out test episodes (10,665 rows). The probe command took 15.98 s for `b0s` and 14.60 s for `a0q`. The local scores are one seed each; do not treat them as local means or standard deviations.

| Model | Best-checkpoint SHA-256 | Probe output |
| --- | --- | --- |
| LeWM `b0s` | `a96004390ebdc072b9d39b0ab2a7b78b6e2941c6a4355b91bd45d44c3c926a43` | `/tmp/codex-lehami-20261014/lehami/experiments/results/k0_probe/b0s_s0/` |
| LeHami `a0q` | `6e04d6daf6fc21c4099fecb256e80a1df4cc01a2c37b758631c1e138876dba05` | `/tmp/codex-lehami-20261014/lehami/experiments/results/k0_probe/a0q_s0/` |

## One-epoch pipeline checks

The official split contains 630 train, 135 validation, and 135 test episodes. The official 25-NN path fit on 49,770 training rows and evaluated on 10,665 held-out test rows. These smoke values validate the training and probe path only. One epoch is not the paper's 35-epoch setting.

| Model | Seed | Smoke training time | Test MAE (rad/s) | Use |
| --- | ---: | ---: | ---: | --- |
| LeHami `a0q` | 0 | About 200 s | 0.863241 | Pipeline check only |
| LeWM `b0s` | 0 | 355.90 s | 2.503771 | Pipeline check only |

The LeHami smoke runner proceeded into unrelated MLP and K=1 probes after saving the official KNN probe. That extra path was stopped. The scoped K=0 calculation then completed using the official evaluation helpers. The LeWM smoke used the official training entry point and the official K=0 KNN helper directly.

The reusable [K=0 probe script](../../scripts/reproductions/lehami_k0_probe.py) calls the official feature collector and KNN helper. It checks the HDF5 hash against the official manifest before scoring and saves the metric, fitted probe, and held-out predictions under the supplied output directory. It was run on both smoke checkpoints; both matched the manifest SHA-256 above.

## Sources and data

- Paper: [LeHami](https://arxiv.org/abs/2610.15267), Table 1.
- Official source: [rominay/LeHami](https://github.com/rominay/LeHami), commit `f76f65f8b99afb6b7e3063f573645d3e2f290ede`, Apache-2.0.
- Baseline source: [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, MIT.
- Dataset: 900 generated 64×64 RGB episodes, 80 frames per episode, with the official seed-42 episode manifest. The HDF5 SHA-256 is `9ae8f830c1ef3e5f4a04414de9278a295817546bfd4f1b9eb4c6bfc19eafae37`.
- Model configs: `a0q` for LeHami and `b0s` for LeWM. Both specify 35 epochs and batch size 32.

## Runtime and deviations

Training runs use a separate checkout and environment under `/tmp`; no research data or model checkpoints are stored in the project. The checkout uses the exact LeHami revision above and the pinned LeWorldModel commit.

The host has an RTX 5070 with 12 GB VRAM. The runtime is Python 3.13.14, PyTorch 2.12.0+cu130, and CUDA runtime 13.0. The author's Python 3.14 lock resolution could not use the locked Box2D wheel; Python 3.13 also exposed an unused `labmaze`/Bazel dependency through the optional `env` extra. The training runtime therefore uses a derived lock with only that unused extra removed. All 223 retained package versions match the official lock. This is a documented environment deviation; the official source files and lock remain unchanged.

The official trainer logged that the split manifest has `num_steps=16` while the selected configs use 11 steps for `b0s` and 14 steps for `a0q`. It accepts the mismatch and trains with each config's window length. Episode membership still comes from the same 630/135/135 episode manifest, and the dataset frame cap remains 80. The trainer logged these expected config values for both full runs. It completed 35 configured epochs (epoch indices 0–34) for each model. `/usr/bin/time -p` recorded the training wall times above.

## Seed-0 K=1-fitted rollout probe

The paper's Figure 15 studies held-out angular-velocity error over an actuated-pendulum rollout. This local check used the pinned official standard evaluator, the existing seed-0 checkpoints, the same generated data and episode split, recorded actions, a 25-neighbor probe fit on K=1 training predictions, and a horizon of 76. The fitted probe stayed fixed at each rollout step. No model training ran in this evaluation.

The table reports `mae_omega__knn_k1` from each `probe_metrics.csv`, using the 135-episode test split. It gives the local seed-0 value in rad/s at selected horizons. The full CSV has 77 rows per split, K=0 through K=76, and the raw output directories also include the evaluator's probe files, rollout metrics, summary, and energy-bin files.

| Rollout step K | LeHami `a0q` test MAE (rad/s) | LeWM `b0s` test MAE (rad/s) |
| ---: | ---: | ---: |
| 1 | 0.467665 | 0.513117 |
| 5 | 0.606430 | 0.811439 |
| 10 | 1.033282 | 0.869993 |
| 20 | 1.393730 | 1.232498 |
| 30 | 1.640654 | 1.481048 |
| 50 | 2.188266 | 1.666773 |
| 76 | 2.881099 | 2.552452 |

For this one seed, LeHami has lower velocity MAE at K=1 and K=5. LeWM has lower MAE at the selected horizons from K=10 onward. This is a descriptive single-seed result. It is not a three-seed estimate and does not reproduce Figure 15. In particular, it does not show that symplecticity guarantees a more accurate long rollout.

### Run conditions and outputs

- Official LeHami source revision: `f76f65f8b99afb6b7e3063f573645d3e2f290ede`.
- Checkpoints: the existing `a0q` seed-0 SHA-256 is `6e04d6daf6fc21c4099fecb256e80a1df4cc01a2c37b758631c1e138876dba05`; `b0s` seed-0 SHA-256 is `a96004390ebdc072b9d39b0ab2a7b78b6e2941c6a4355b91bd45d44c3c926a43`.
- Dataset: both `STABLEWM_HOME/pendulum_swingup.h5` files match the manifest SHA-256 `9ae8f830c1ef3e5f4a04414de9278a295817546bfd4f1b9eb4c6bfc19eafae37`. The manifest has 630 train, 135 validation, and 135 test episodes; the evaluator used the 630 train and 135 test episodes.
- Runtime: Python 3.13.14, PyTorch 2.12.0+cu130, CUDA 13.0, h5py 3.16.0, scikit-learn 1.8.0, and NVIDIA GeForce RTX 5070 (12,227 MiB reported memory).
- The official runner also fit its other standard probes and wrote energy-stratified metrics. These are retained in the raw outputs; the table above uses only the K=1-fitted 25-NN probe and test rows.
- Deviation from the paper: this check uses one training seed and rollout seed 0 per model, not the paper's three architecture seeds. The official evaluator, data split, recorded actions, probe size, and horizon were not changed.
- Wall time: `a0q` 1,149.88 s (19m09.88s); `b0s` 1,387.59 s (23m07.59s), measured with `/usr/bin/time -p`.
- Raw `a0q` output: `/tmp/codex-lehami-20261014/lehami/experiments/results/codex_rollout_a0q_seed0_20261015/`.
- Raw `b0s` output: `/tmp/codex-lehami-20261014/lehami/experiments/results/codex_rollout_b0s_seed0_20261015/`.
- Logs and time files: `/tmp/codex-lehami-20261014/lehami/experiments/logs/actuated_rollout/`.

The commands used were:

```sh
STABLEWM_HOME=/tmp/lehami-full-a0q-seed0-20261014-home \
  /tmp/lehami-runtime-env/.venv/bin/python \
  /tmp/codex-lehami-20261014/lehami/experiments/src/run_eval.py \
  --checkpoint /tmp/lehami-full-a0q-seed0-20261014-home/checkpoints/lehami_sg_sep_sreg_q_ms_conv_pendulum_swingup_lehami_sg_sep_conv_w1_swingup_exp079_a0q_s0/best.ckpt \
  --manifest /tmp/codex-lehami-20261014/lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json \
  --exp-name codex_rollout_a0q_seed0_20261015 \
  --horizon 76 --device cuda --knn-neighbors 25 --rollout-seed 0

STABLEWM_HOME=/tmp/lehami-full-b0s-seed0-20261014-home \
  /tmp/lehami-runtime-env/.venv/bin/python \
  /tmp/codex-lehami-20261014/lehami/experiments/src/run_eval.py \
  --checkpoint /tmp/lehami-full-b0s-seed0-20261014-home/checkpoints/lewm_ms_conv_pendulum_swingup_lewm_ms_conv_w1_swingup_exp072_b0s_s0/best.ckpt \
  --manifest /tmp/codex-lehami-20261014/lehami/data/split_pendulum_swingup_episode_srnn_K80_val.15_seed42.json \
  --exp-name codex_rollout_b0s_seed0_20261015 \
  --horizon 76 --device cuda --knn-neighbors 25 --rollout-seed 0
```

## Reproduction limits

- One seed per model cannot reproduce a three-seed estimate or its variability. The local `a0q` score is 0.456258 rad/s; the local `b0s` score is 2.504686 rad/s. The paper reports three-seed means and standard deviations, so this single-seed comparison does not establish statistical agreement or disagreement.
- The K=1-fitted rollout curves also use only seed 0. They add measured prediction evidence, but do not provide a local multi-seed summary or establish a planning or control benefit.
- The smoke runs use one epoch and are not comparable to the paper's full training recipe.
- The selected metric is an offline held-out feature probe. It does not show that the model supports better plans or control.
- The result applies to this generated pendulum dataset, split, probe, and configs. It does not establish general physical validity.

Keep seeds 1 and 2 pending. Do not report a local three-seed mean or standard deviation until all six runs and probes are complete.
