# Quickstart: Bounded LpWM PushT Reproduction

This guide defines one partial reproduction attempt. It needs about 2.8 GB for the archive, additional space for extraction and checkpoints, and a CPU environment compatible with the upstream code. A local run on a 24-thread CPU host was OOM-killed while loading the full training set with the default 20 data-loader workers. Do not start the full pair on a similarly constrained host without a safe memory plan. The data source has no license metadata, so the data must remain local and must not be redistributed.

## 1. Prepare temporary paths

Use the reviewed upstream revision from [research.md](research.md). Keep the checkout, environment, data, logs, and checkpoints under `/tmp`.

```bash
git clone https://github.com/YilunKuang/lpworldmodel.git /tmp/lpwmworldmodel
git -C /tmp/lpwmworldmodel checkout bdd812d9432cccda8c350086006401b436f91982
uv python install 3.9.25
uv venv --python 3.9.25 /tmp/lpwm-venv
```

Install the upstream requirements in this environment. Use CPU wheels for PyTorch 2.3.0 and torchvision 0.18.0. Exclude CUDA-only packages from the author's environment file. Keep all other upstream package versions fixed where possible and record any install changes in the final report.

## 2. Verify and extract the dataset

Download `pusht_noise.zip` from the OSF file link in [research.md](research.md). Before extraction, confirm the archive has the expected byte count and SHA-256 listed there. Stop if either check fails. Extract it so `DATASET_DIR` contains the `pusht_noise/` directory expected by upstream scripts.

## 3. Check the selected upstream cells

From the upstream checkout with the isolated environment active:

```bash
ARM_LIST="sparse dense" PRED_LIST=ltv PD_LIST=768 bash scripts/reproduce_pusht.sh
```

Confirm this dry run prints exactly two training and evaluation cells. Run a small CPU preflight and confirm the data paths and selected model import work before starting training.

## 4. Train and evaluate

Set `DATASET_DIR` to the extracted data root and `CKPT_BASE=/tmp/lpwm-runs`. Then run the filtered author script with `RUN=1`, the same two-cell filters, and the settings in [plan.md](plan.md). The script evaluates seed 99. For every checkpoint that completes, run `scripts/plan.sh` with `SEED=100` and then `SEED=101`, preserving the same planner configuration and 50-episode count.

Save command output and raw metrics under `/tmp`. Record failures and incomplete episode counts. Do not tune settings against these evaluation results.

## 5. Report

Use the fields in [data-model.md](data-model.md) and the project [evaluation protocol](../../docs/EVALUATION_PROTOCOL.md). Report each planner seed, mean, standard deviation, and sparse-minus-dense difference. Apply the decision rule written in [research.md](research.md). Label the result a partial reproduction attempt, even if the direction matches.
