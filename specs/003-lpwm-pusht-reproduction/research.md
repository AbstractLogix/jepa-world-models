# Research: Bounded LpWM PushT Reproduction

**Research date**: 2026-10-14

## Selected claim

The LpWM paper reports that sparse LpWM improves PushT planning success over dense LeWM for intermediate-capacity predictors. For MLP∘LTV predictors, it reports a 36%–45% success-rate advantage across the tested latent dimensions. This feature checks only the authors' D=768 configuration. It cannot reproduce the full range or the paper's maximum reported advantage.

Primary source: [LpWM paper](https://arxiv.org/abs/2608.22764), especially Section 3 and Appendix H.1.

## Author code and configuration

- Official repository: [YilunKuang/lpworldmodel](https://github.com/YilunKuang/lpworldmodel)
- Reviewed revision: `bdd812d9432cccda8c350086006401b436f91982` (2026-08-26).
- Official grid script: [scripts/reproduce_pusht.sh at the reviewed revision](https://github.com/YilunKuang/lpworldmodel/blob/bdd812d9432cccda8c350086006401b436f91982/scripts/reproduce_pusht.sh).
- A local dry run with `ARM_LIST="sparse dense" PRED_LIST=ltv PD_LIST=768` emitted exactly two cells. It did not train or evaluate a model.
- The sparse LpWM D=768 setting uses `link=reprelu`, `target_p=1`, `mu=0`, `REG_WEIGHT=1.0`, and `MUP_LR=5e-4`.
- The dense LeWM D=768 setting uses `link=identity`, `target_p=2`, `REG_WEIGHT=10.0`, and `MUP_LR=5e-5`.
- Both author commands use `MUP=1`, a scratch ViT CLS encoder, `frameskip=5`, `num_hist=3`, 2 epochs, and batch size 64.
- The official `train_rdmreg.yaml` defaults to training seed 0. The paper states one training seed but does not name its value in the prose.

The full author script defaults to 2 arms × 6 predictor families × 5 latent dimensions (60 sequential cells). The filtered D=768 MLP∘LTV attempt covers only one pair. The official script performs one planner run per cell; its planner config defaults to seed 99. The paper reports three planning seeds but does not name them. The local attempt will keep seed 99 and add seeds 100 and 101, and will report those choices as local settings.

## Planning protocol

The selected author config uses closed-loop CEM-MPC with a goal horizon of 5, 50 evaluation trajectories, at most 10 MPC replanning iterations, 5 executed actions per iteration, and a CEM subplanner with 300 samples, top-k 30, and 30 optimization steps. The metric is the percentage of evaluation trajectories that reach the goal. Report success by seed, then the mean and standard deviation across seeds for each model arm.

The paper's three-seed planning mean is not an exact seed-for-seed target because its prose does not identify the seed values. The local values must remain separate from the paper's plot and range.

## Decision rule set before local evaluation

For the selected D=768 configuration, the directional claim is supported by this attempt only if the sparse arm's mean success rate is greater than the dense arm's mean over the three local planner seeds (99, 100, 101). Report all seed values and the difference. This rule does not establish statistical significance, test the exact paper value at D=768, or reproduce the paper's full 36%–45% range.

## Dataset and access

- The author repository identifies the source as the [DINO-WM dataset](https://github.com/gaoyuezhou/dino_wm), hosted in the [public OSF project](https://osf.io/bmw48/).
- The required archive is `pusht_noise.zip`, listed by the [OSF API](https://api.osf.io/v2/nodes/bmw48/files/osfstorage/).
- OSF metadata reports 2,785,304,515 bytes and SHA-256 `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08`.
- The OSF node is public, but the API response contains no license metadata. This is an access fact, not evidence of usage or redistribution rights. Keep the archive local and do not redistribute it.
- The local machine had 763 GB free when checked. The archive was downloaded, verified, and extracted under `/tmp/lpwm-data`, outside the repository.

## Runtime and deviations

- The author's `environment.yaml` specifies Python 3.9, PyTorch 2.3.0, and torchvision 0.18.0, along with CUDA packages. Keep this environment isolated from the tutorial project's Python environment.
- This host has no CUDA device. The selected training path uses Accelerate's device selection and the planning entry point selects CPU when CUDA is unavailable. Run a CPU preflight before training; stop if the selected path requires CUDA.
- Use CPU PyTorch wheels while retaining the upstream PyTorch and torchvision versions. Record the CPU wheel choice as a deviation from the CUDA environment file.
- Keep logs, checkpoints, and extracted data under `/tmp/lpwm-data` and `/tmp/lpwm-runs`. Do not copy author code or model outputs into the tutorial repository.

## Alternatives considered

- Running the full 60-cell grid would better reproduce Figure 1(b), but it would train and evaluate 60 cells sequentially. This first attempt is bounded to one paired setting and must be labeled partial.
- Using a different predictor or a new local hyperparameter sweep would not test the selected author configuration. Use the exact D=768 `ltv` settings from the pinned author script.
- Reusing published checkpoints would skip the training comparison. Train both arms from scratch if the CPU preflight and run complete.
