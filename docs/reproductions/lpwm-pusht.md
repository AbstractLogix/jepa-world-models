# LpWM PushT D=768 Reproduction Attempt

**Attempt date**: 2026-10-14

**Status**: Partial reproduction attempt; incomplete. Training stopped before the first epoch completed. No planner run or paper result was reproduced.

## Selected claim

The [LpWM paper](https://arxiv.org/abs/2608.22764) reports higher PushT planning success for sparse LpWM than dense LeWM at intermediate predictor sizes. For MLP∘LTV, it reports a 36%–45% advantage across tested sizes. This attempt selected the authors' D=768 setting. It did not test that full range.

Before evaluation, the local decision rule was set: the claim is supported at D=768 only if sparse LpWM has a higher mean success rate than dense LeWM over local planner seeds 99, 100, and 101. This rule was not applied because no model completed training.

## Sources and data

| Item | Value |
| --- | --- |
| Paper | [LpWM, arXiv:2608.22764](https://arxiv.org/abs/2608.22764) |
| Author code | [YilunKuang/lpworldmodel](https://github.com/YilunKuang/lpworldmodel), revision `bdd812d9432cccda8c350086006401b436f91982` |
| Dataset source | [DINO-WM OSF project](https://osf.io/bmw48/), file `pusht_noise.zip` |
| Expected archive size | 2,785,304,515 bytes |
| Expected and observed SHA-256 | `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08` |
| Data terms | OSF node is public. Its API has no license metadata. Use and redistribution terms remain unknown. |
| Local data | Downloaded and extracted under `/tmp/lpwm-data`; no data was added to this repository. |

The archive matched both published checks. A loader check read one train and one validation sample. The reported dataset splits contained 18,685 train rollouts and 21 validation rollouts.

## Intended comparison

Both arms use the same dataset, scratch ViT CLS encoder, frameskip 5, history 3, two epochs, batch size 64, seed 0, and MLP∘LTV predictor at D=768. These settings came from the pinned author script.

| Arm | Link | Target p | mu | Regularization weight | muP learning rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Sparse LpWM | reprelu | 1 | 0 | 1.0 | 5e-4 |
| Dense LeWM | identity | 2 | not set | 10.0 | 5e-5 |

The planned CEM-MPC evaluation uses 50 trajectories for each planner seed 99, 100, and 101. The settings are goal horizon 5, at most 10 replanning steps, 5 executed actions per step, 300 CEM samples, top-k 30, and 30 optimizer steps. The upstream script would run seed 99 after each training run; seeds 100 and 101 require separate planner calls.

## Local attempt

The filtered upstream dry run printed exactly the selected two cells. The runtime used Python 3.9.25, PyTorch 2.3.0 CPU, torchvision 0.18.0 CPU, Accelerate 0.26.1, W&B 0.13.1, and setuptools 69.5.1 on an AMD Ryzen 9 5900X host with 24 logical CPUs. CUDA was unavailable. Setuptools was pinned in the temporary environment to restore `pkg_resources`, which the upstream W&B version imports.

W&B offline mode failed at startup with `BrokenPipeError` while setting the run name. The same error occurred in a small standalone check. W&B disabled mode passed its startup check and let the training process reach the dataset loader. This disabled experiment tracking for the local attempt; it did not change model settings.

The sparse arm loaded 18,685 rollouts and started epoch 1. The progress bar reported 30,965 batches in that epoch. The process ended with exit status 137 before the epoch completed. The Linux cgroup `oom_kill` counter increased. `/usr/bin/time` reported 53.73 seconds elapsed and 6,371,596 KiB maximum resident set size for the main process. This does not include a measured peak for all data-loader workers.

The cause of the memory kill is not known. The upstream training script used its default of 20 data-loader workers. The high worker count and loaded video data may have increased total memory use, but that is an inference, not a measured cause.

| Run | Outcome |
| --- | --- |
| Sparse LpWM training | Started; stopped during epoch 1; no completed checkpoint |
| Dense LeWM training | Not started |
| Planner seeds 99, 100, 101 | Not run; no trained checkpoints |
| Success rates, means, standard deviations | Unavailable |

Raw logs and the failed temporary run directory remain under `/tmp/lpwm-runs-disabled`. No checkpoint was produced. No value was entered as a zero score.

## Conclusion and limits

This is an **incomplete reproduction attempt**, not a reproduced finding or a performance comparison. The pre-run decision rule remains unevaluated. The paper's reported range is still paper-reported only.

The full author setting requires 30,965 batches per epoch for each arm. At two epochs and two arms, that is about 123,860 training batches, before six planner runs. The CPU attempt also ended with an out-of-memory kill. Do not continue this exact run on the current host without a safe resource plan.

A GPU-capable host can continue the same data and model setting. A CPU run with fewer data-loader workers may reduce memory pressure, but it would still need the full training budget and must record the worker change. A smaller data subset would be a scaled experiment and could not validate this paper claim.

The paper, author code, OSF data record, exact configuration, checksum, and failure evidence are recorded. Data, checkpoints, and raw outputs remain outside version control. The report does not claim any data-use or redistribution rights.
