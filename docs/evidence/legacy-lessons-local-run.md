# Local Evidence for Notebooks 01 and 04–08

**Run date**: 2026-10-14  
**Run status**: Complete for the listed local CPU protocols  
**Evidence files**: `/tmp/jepa-local-evidence-20261014/executed-final/`

This report records local notebook execution. These lessons use toy data, synthetic targets, or small control environments. Their outputs do not reproduce paper benchmarks or validate paper findings.

## Environment

| Item | Measured value |
| --- | --- |
| Python | 3.13.14 |
| PyTorch / TorchVision | 2.10.0+cpu / 0.25.0+cpu |
| Gymnasium | 0.29.1 |
| timm | 1.0.30 |
| NumPy / pandas | 2.5.4 / 2.3.3 |
| CPU | AMD Ryzen 9 5900X, 12 cores / 24 logical CPUs |
| System memory | 31 GiB |
| GPU present | NVIDIA GeForce RTX 5070, 12,227 MiB |
| GPU used | No. PyTorch reported `cuda_available=False`; every run used CPU. |

## Run summary

| Notebook | Clean final run | Runtime | Seeds and measured result | Boundary |
| --- | --- | ---: | --- | --- |
| 01 | `run-f/01_intuition.ipynb` | 20.84 s | NumPy, PyTorch, Swiss-roll, and generated-view seeds are 42. Each of the three toy variants ran 1,500 epochs. Final logged loss rounded to 0.0000 for the naive and VICReg-style variants; the LeJEPA-inspired isotropic proxy ended at 0.2256. | Synthetic 2-D concept demo. The proxy is not the paper's SIGReg objective. |
| 04 | `run-f/04_vjepa_plus_v1_jepa.ipynb` | 25.59 s | Data/model seeds 42; 1,200 CIFAR-10 images split into 960 train and 240 held-out. Held-out cosine alignment 0.615; descriptor Recall@1 0.021. | Uses pinned `timm/resnet18.a1_in1k` revision `491b427b45c94c7fb0e78b5474cc919aff584bbf` (Apache-2.0; weights SHA-256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`). Targets are synthetic brightness and edge-strength descriptors, not text. This is not V-JEPA or VL-JEPA. |
| 05 | `run-c/05_value_guided_planning.ipynb`, repeated in `run-d/05_value_guided_planning.ipynb` | 30.59 s / 29.77 s | Training reset/action seeds 42–121. Both runs used evaluation seeds 100–124. Random-policy returns in both runs: `[19,49,18,63,38,15,19,29,23,35,12,29,37,19,27,30,34,19,22,16,20,21,26,19,11]`; mean 26.00. Planned-policy returns in both runs: `[199,234,215,184,201,196,189,213,179,228,236,188,212,173,199,215,248,211,203,248,193,226,189,186,270]`; mean 209.40. Both complete return lists matched across runs. | CartPole latent-dynamics control analogue, not a video world model or robot result. Returns from one trained model do not estimate training-seed uncertainty. |
| 06 | `run-c/06_gradient_based_planning.ipynb`, repeated in `run-d/06_gradient_based_planning.ipynb` | 280.15 s / 268.85 s | Evaluation reset/candidate seeds 100–119. Both runs used random-shooting returns `[213,247,219,187,205,196,200,201,202,230,239,188,211,174,200,227,247,229,204,277]` (mean 214.80) and gradient-planning returns `[92,75,78,119,97,108,100,88,102,74,76,105,79,130,92,84,70,82,101,67]` (mean 90.95). Both complete return lists matched across runs. | CartPole planner example. This does not reproduce the GRASP paper. The measured outcome is specific to this notebook's model, planner, and seeds. |
| 07 | `run-a/07_latent_action_world_models.ipynb` | 11.96 s | Data generation begins at seed 42; logged episode 529 uses reset/action seed 571. The notebook reports a training-set one-step latent MSE of 0.004930 on the first 2,000 of 12,000 transitions. Code usage: `[0.120,0.130,0.146,0.110,0.096,0.134,0.104,0.159]`. | CartPole training-set metric. It is not a held-out score or a result for in-the-wild video. |
| 08 | `run-e/08_physical_planning_ablation.ipynb` | 8.30 s | Training seed 42; shared evaluation seeds 200–209. See the complete outcomes below. | One training seed per setting. Episode standard deviation is not training-seed uncertainty. This is a CartPole ablation, not a physical-planning reproduction. |

### Notebook 08 per-setting outcomes

The reported standard deviation is across ten episode returns from one trained model.

| Latent size | Predictor depth | Value weight | Episode returns | Mean | Episode SD |
| ---: | ---: | ---: | --- | ---: | ---: |
| 32 | 3 | 0.20 | `[12,10,13,11,10,10,10,10,12,10]` | 10.8 | 1.077033 |
| 32 | 3 | 0.05 | `[9,10,15,9,10,11,10,12,11,10]` | 10.7 | 1.676305 |
| 64 | 3 | 0.05 | `[12,10,13,11,10,10,8,9,13,10]` | 10.6 | 1.562050 |
| 64 | 3 | 0.20 | `[11,9,13,10,9,10,8,10,12,9]` | 10.1 | 1.445683 |
| 32 | 1 | 0.20 | `[8,9,15,8,10,10,10,10,10,10]` | 10.0 | 1.843909 |
| 32 | 1 | 0.05 | `[8,9,10,8,10,10,10,10,9,10]` | 9.4 | 0.800000 |
| 64 | 1 | 0.20 | `[11,9,9,10,9,9,8,8,10,9]` | 9.2 | 0.871780 |
| 64 | 1 | 0.05 | `[11,9,9,10,9,9,8,8,10,9]` | 9.2 | 0.871780 |

## Failures fixed before the final runs

- The first Notebook 05 evaluation spent more than six minutes in scalar candidate scoring and was interrupted before the planned-policy episodes completed. Batched scoring reduced each final run to about 30 seconds while keeping the seeded candidate sequences.
- An earlier Notebook 06 run failed after reusing a computation graph during gradient planning. Detaching the fixed initial state resolved the error. Two complete runs then produced identical returns.
- An earlier Notebook 08 run failed because the plot requested `reward_std` while the results table used `episode_reward_std`. The plot now reads the existing field; the clean run completed.
- The final executed copies for all six notebooks contain no cell errors. The earlier failures remain part of this report so the repair path is clear.

## Reproduction and claim limits

- Notebook runs use the same seeded inputs within each comparison. Exact repeated returns were confirmed for notebooks 05 and 06 in two separate run roots on this CPU environment.
- This check does not prove bitwise repeatability on other software, operating systems, or hardware.
- The local objectives, datasets, and environments are reduced teaching examples. Do not compare these values with paper tables or cite them as confirmation of a research claim.
- Notebook 04's held-out retrieval is against synthetic two-value descriptors. It is not natural-language retrieval.
- Notebook 07 reports a training-set metric. Notebook 08 reports episode spread for one trained model. Neither is a held-out or multi-training-seed estimate.
- The longer Notebook 06 runtime is CPU planning time for this local setup; no GPU performance claim is made.
