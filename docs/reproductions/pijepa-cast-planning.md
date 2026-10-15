# PiJEPA CAST Table 2: Preflight Report

**Reviewed**: 2026-10-15  
**Status**: Not run; preflight blocked  
**Scope**: V-JEPA-2 ViT-L; PiJEPA, Octo-WM, Octo, and default MPPI; CAST validation cases

## Paper target

[Policy-Guided World Model Planning for Language-Conditioned Visual Navigation](https://arxiv.org/html/2603.25981v1), Table 2, reports ATE XY RMSE of 1.65 m for PiJEPA, 1.67 m for Octo-WM, 1.72 m for Octo, and 1.87 m for default MPPI. These values are paper results. No local PiJEPA measurements exist.

## Preflight findings

| Gate | Evidence checked | Status |
| --- | --- | --- |
| First-party code | [PiJEPA revision `f38a265cbce4fac5aa5fd9b9f7b18c723bccb645`](https://github.com/AmirhoseinCh/PiJEPA/tree/f38a265cbce4fac5aa5fd9b9f7b18c723bccb645). Rechecked 2026-10-15: `main` still points to this commit. Root license is MIT. README specifies Python 3.10 and four distributed processes. The tracked tree has no policy or world-model weights. | Current source is pinned. Exact method checkpoints remain unverified. |
| Dataset terms | [CAST dataset card at `fe9de38e1e519e649a40db20c9558f94e52887a0`](https://huggingface.co/datasets/catglossop/CAST-dataset/tree/fe9de38e1e519e649a40db20c9558f94e52887a0). Rechecked 2026-10-15: `main` still points to this commit. The card lists 80.8 GB, has empty metadata, and states no license or usage terms. It lists seven GNM source datasets. | **Blocked. No data downloaded.** Get authoritative terms for the released data and its source mixture before transfer. Public access does not establish permission. |
| Required checkpoints | PiJEPA's public tracked tree has no policy or world-model weights; README paths point outside the repository. A separate [CounterfactualVLA checkpoint](https://huggingface.co/catglossop/CounterfactualVLA/tree/5d18524b39a844d245b497ddc596a559c07d8bec) is public at 32.4 GB, but its record does not map it to the four Table 2 assets or state its license. The [V-JEPA-2 ViT-L encoder](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f) is available at revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f` under MIT; it is not a trained policy or world model. | **Blocked. No model downloaded.** The separate CAST checkpoint is not a verified substitute. |
| Evaluator and cases | Rechecked at the current PiJEPA revision. The wrapper defaults to 1,000 examples and seed 42, sets MPPI to 4 iterations, 32 samples, 4 elites, temperature 0.8, and standard-deviation bounds 0.01–0.05, but defaults `N_OCTO_SAMPLES` to 1. The paper states four Octo diffusion samples (`N_pi=4`) for its policy prior. The evaluator applies one sample-count setting across modes, and `octo_mean` averages all generated samples; the paper does not state the sample count for standalone Octo. The modes map as `default_mppi` = Default MPPI, `octo_mppi` = PiJEPA, `octo_wm` = Octo-WM, and `octo_mean` = Octo. The evaluator passes the raw policy-sample standard deviation into the first policy-seeded MPPI iteration without clipping it to the paper's bounds; later MPPI updates are clipped. Validation ordering is deterministic, but the paper does not publish the exact evaluation-case manifest. | **Blocked.** Confirm sample handling for standalone Octo before choosing per-method invocations. Resolve or record the initial-standard-deviation difference and retain the ordered case list and hash; a case match remains unconfirmed unless authors verify it. |
| Compute and storage | The paper reports four H200 GPUs. Current host has one RTX 5070 with 12 GB VRAM. CAST archives are 80.8 GB before expansion. | **Blocked.** Current host is not qualified; no transfer or run started. |
| Run count | Paper does not state the training seed count. Project protocol requires five independent seeds for stochastic training when the paper gives no run count. | Pending future supported host and assets. A smaller run stays exploratory or partial. |

The [PiJEPA repository](https://github.com/AmirhoseinCh/PiJEPA) has an MIT root license, while the bundled [JEPA-WMS component](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0) is CC BY-NC 4.0. Treat those as separate license scopes. The encoder's license does not settle CAST data terms or grant access to PiJEPA-trained checkpoints.

## Work performed

Source revisions and public asset listings were rechecked on 2026-10-15. Only the pinned public source-code revision was cloned for inspection. No CAST archive, model weight, or checkpoint was downloaded. No training or evaluator run was started. The arXiv record still lists v1 only. This report records preflight evidence only; it does not reproduce a paper finding.

## Next steps

1. Find and verify authoritative published CAST usage terms and record their scope; stop if owner input is required.
2. Verify first-party access to every required method-specific checkpoint and record each source and hash.
3. Recheck all code, data, and model revisions and license terms.
4. Secure a compatible compute and storage environment, including durable reviewer access for raw evidence stored outside Git.
5. Pin the validation-case list and its hash. Resolve the policy-sample count by method, including the paper's four samples for the PiJEPA prior and the unspecified Octo baseline count. Resolve or document the initial-standard-deviation difference and pin the four-method invocation.
6. Only after every gate passes, run all planned independent seeds and retain raw per-case outputs before comparing local values with the paper.

The detailed plan is in [feature 017](../../specs/017-pijepa-cast-reproduction/spec.md). Keep this status at `Not run` until each gate passes.
