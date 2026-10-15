# Research: PiJEPA CAST Table 2

**Reviewed**: 2026-10-15

## Decision

Target the V-JEPA-2 ViT-L results in Table 2 of [Policy-Guided World Model Planning for Language-Conditioned Visual Navigation](https://arxiv.org/html/2603.25981v1). Compare PiJEPA, Octo-WM, Octo, and default MPPI on the same CAST validation cases. Use ATE XY RMSE in meters as the primary metric and retain the other Table 2 metrics returned by the official evaluator.

The published ATE XY RMSE values are PiJEPA 1.65 m, Octo-WM 1.67 m, Octo 1.72 m, and default MPPI 1.87 m. The paper reports four H200 GPUs. These are paper values, not measurements from this repository.

## Source and artifact snapshot

| Input | Reviewed source | Finding | Run gate |
| --- | --- | --- | --- |
| PiJEPA code | [Official repository at `f38a265`](https://github.com/AmirhoseinCh/PiJEPA/tree/f38a265cbce4fac5aa5fd9b9f7b18c723bccb645), rechecked 2026-10-15 | The `main` branch still points to this revision. Root license is MIT. README requires Python 3.10 and four distributed processes for training/evaluation, and names checkpoint paths outside the repository. The tracked tree has no policy or world-model weights. | Access to the exact PiJEPA and comparison checkpoints remains unverified. The root MIT license does not grant rights to separate data or bundled components. |
| CAST data | [Dataset card](https://huggingface.co/datasets/catglossop/CAST-dataset/tree/fe9de38e1e519e649a40db20c9558f94e52887a0), rechecked at revision `fe9de38e1e519e649a40db20c9558f94e52887a0` | The `main` branch still points to this revision. The card lists 80.8 GB across filtered, counterfactual, and atomic archives. Its metadata is empty and the repository has no license file or dataset usage terms. It says the data draws on seven GNM source datasets. | **Blocked.** Obtain and record authoritative terms for the released data and its source mixture before any download. Public availability is not permission. |
| V-JEPA-2 ViT-L weights | [Official model card at revision `b3c1679`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f) | The current `main` revision is `b3c1679b7c34d3255ef3547f27c7b226aefab26f`; the card reports MIT licensing and lists a 1.3 GB encoder weight file. This is an encoder, not a PiJEPA-trained policy or world-model checkpoint. No model file was downloaded or hashed locally. | Record this revision as the encoder candidate. Before a run, verify that it matches the paper's training configuration and hash the exact downloaded file. |
| Bundled JEPA-WMS component | [Official repository at `13cf1d9`](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0) | The `main` branch still points to this revision. Its bundled license is CC BY-NC 4.0, separate from PiJEPA's root MIT license. | Review license scope and intended use before reusing or transferring this component. |
| Evaluation wrapper | [`run_eval_dino_vjepa.sh` in PiJEPA](https://github.com/AmirhoseinCh/PiJEPA/tree/f38a265cbce4fac5aa5fd9b9f7b18c723bccb645) | The pinned source is still current. The wrapper defaults to 1,000 examples and seed 42, and sets MPPI to 4 iterations, 32 samples, 4 elites, temperature 0.8, and standard-deviation bounds 0.01–0.05. Its default planner list selects `octo_mean`, not all four methods; the explicit modes are `default_mppi` (Default MPPI), `octo_mppi` (PiJEPA), `octo_wm` (Octo-WM), and `octo_mean` (Octo). The paper uses four Octo diffusion samples (`N_pi=4`) for the policy prior, while the wrapper defaults `N_OCTO_SAMPLES=1`. The evaluator applies this value across modes, and `octo_mean` averages all generated samples. The paper does not state the sample count for standalone Octo, so one shared value cannot be assumed to match every method. The evaluator passes the raw sample standard deviation to the first policy-seeded MPPI iteration without clipping it to the paper's stated bounds; it clips only later updates. The inner evaluator defaults to 50 examples. Validation ordering is deterministic, but the paper publishes no exact case list. | The initial-standard-deviation mismatch and missing case manifest must be resolved or recorded as deviations before a run can support a paper-aligned claim. Confirm sample handling for the standalone Octo baseline before selecting per-method invocations. Preserve the ordered case list and per-case JSON. Do not use the inner evaluator defaults by accident. |

The arXiv record still lists only v1 as of 2026-10-15. The CAST authors also publish a separate 32.4 GB [CounterfactualVLA checkpoint](https://huggingface.co/catglossop/CounterfactualVLA/tree/5d18524b39a844d245b497ddc596a559c07d8bec). The public record does not map it to any of the four PiJEPA Table 2 assets or state its license, so it is not treated as a substitute or an approved run asset.

The paper does not state the training seed count or publish the exact validation-case manifest. The official evaluator uses deterministic validation ordering, but that does not prove it is the same case list used for the paper. Unless the authors confirm the case list, a local run must be labeled partial even if its metrics are close.

## Compute and run design

The current host has one RTX 5070 with 12 GB VRAM. The paper reports four H200 GPUs, and the official README configures four-process distributed work. The local host is therefore not qualified for the planned paper-aligned run. Do not start training on this host as a substitute.

The project evaluation protocol requires at least five independent training seeds for stochastic training when the paper gives no run count. Use the paper's count when stated. For frozen checkpoints, record the exact source and hash; evaluation episodes do not replace training seeds. If the official assets or compute support fewer runs, report the exact missing runs and keep the result partial or exploratory.

## Alternatives considered

- **Run only the official wrapper defaults**: rejected because the default planner list does not cover all four compared methods.
- **Use V-JEPA-2 weights as a substitute for missing PiJEPA checkpoints**: rejected because an encoder checkpoint is not the trained policy and world model used by PiJEPA.
- **Download CAST while terms are unknown**: rejected. The dataset card does not state permission.
- **Run a smaller local subset or one seed**: acceptable only as a separately labeled exploratory result after license and compute gates pass. It cannot satisfy this feature's paper-aligned reproduction target.
- **Use a local simulator analogue**: already covered by Notebook 16. Its hand-written controller, CEM, and MountainCar frames do not reproduce PiJEPA's learned-policy MPPI result.

## Stop conditions

Keep the run at `Not run` until all of these are documented in [the preflight report](../../docs/reproductions/pijepa-cast-planning.md): authoritative CAST terms; source and hash for every required checkpoint; exact fixed validation cases and their hash; compatible compute and storage; and a pinned evaluator invocation that selects all four methods. Recheck every revision and license at the time of a future run.
