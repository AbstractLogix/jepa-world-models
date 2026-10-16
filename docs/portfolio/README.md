# Portfolio files

This folder contains the public case study and a JSON manifest for the Abstract Software website.

## Website use

`project.json` is plain JSON. A SvelteKit site can import it as a build-time data module or copy it into the site's project-data folder. Keep the `schemaVersion` field and render only fields the page understands. The manifest uses repository-relative paths for local evidence and media. A site build should copy those files into its static assets or resolve them against the repository URL at a pinned commit.

Example:

```ts
import project from "$lib/data/projects/jepa-world-models.json";
```

Use `media[].alt` as the image alternative text and `media[].caption` as the visible caption. Show each finding's `status`, `scope`, and `claimBoundary` with the summary. Do not present the MountainCar analogues as JEPA results or paper reproductions.

Render `pendingWork` as an open or gated item. Keep it separate from `verifiedFindings`; it describes work that has no completed result to publish.

## Claim checks

| Claim | Source to inspect | Safe wording |
| --- | --- | --- |
| SIGReg probe difference | [Notebook 02 report](../reproductions/lejepa-rectified-lpjepa.md) | Small five-seed CIFAR-10 objective study; interval includes zero. |
| RDMReg projector result | [Notebook 03 report](../reproductions/lejepa-rectified-lpjepa.md) | Local projector-probe increase over a weak teaching proxy; not ImageNet-100. |
| Decision-usefulness contrast | [Notebook 14 report](../reproductions/decision-usefulness.md) and saved notebook outputs | State-model MPC failed on these 20 starts; direct momentum rule succeeded; actual compute differed. |
| Prior-guided search analogue | [Notebook 16 report](../reproductions/action-video-planning.md) and [episode CSV](../evidence/action-video-planning-episodes.csv) | A hand-written prior improved this local CEM setup; it is not a learned policy or JEPA result. |
| Calibration under planner shift | [Notebook 17 report](../reproductions/model-uncertainty-planning.md) and [raw rows](../evidence/model-uncertainty-calibration.csv) | Five state-model fits had wider planner-shift uncertainty; one task only. |
| Official V-JEPA checkpoint execution | [Notebook 09 run record](../evidence/vjepa21-local-run.md) | Checkpoint inference on one synthetic clip; no benchmark or planning claim. |
| Notebook 22 local comparison | [Notebook 22 report](../reproductions/visual-jepa-decision-usefulness.md), [model records](../evidence/visual-jepa-model-runs.csv), [interventions](../evidence/visual-jepa-interventions.csv), and [episodes](../evidence/visual-jepa-episodes.csv) | The VISReg-style adaptation passed its fixed spread screen. Uniform-start JEPA CEM succeeded on 0/100 starts; momentum-prior JEPA CEM succeeded on 100/100, while the direct hand-written prior succeeded on 20/20. Do not attribute the outcome to JEPA alone. |
| Notebook 23 matched study | [v1 study report](../reproductions/visual-jepa-decision-study-v1.md), [frozen protocol](../reproductions/protocols/visual-jepa-decision-study-v1.md), [660 episode rows](../evidence/visual-jepa-decision-study-v1-episodes.csv), [paired analysis](../evidence/visual-jepa-decision-study-v1-analysis.json), and [validator](../../scripts/validate_visual_jepa_decision_study.py) | The matrix is complete and validated. Publish the local success counts and paired analysis with the synthetic task, privileged-state, compute, gate, paired-seed, and low-spread limits. Keep the claim partial and JEPA outcomes exploratory. Prior-only CEM and a learned direct policy were not run. |

“Verified” in the manifest means the value was checked against local records. It does not mean an independent party repeated the experiment. PRs #1–#4 are merged into `main`; this case study uses a separate draft PR.

## Rebuild and validate

From the repository root:

```bash
uv sync --locked
uv run --locked --no-sync python scripts/generate_portfolio_figures.py
uv run --locked --no-sync python scripts/analyze_visual_jepa_decision_study.py
uv run --locked --no-sync python scripts/generate_visual_jepa_decision_study_figures.py
uv run --locked --no-sync python -m unittest discover -s tests
```

The original figure script reads the saved Notebook 14 episode table and retained CSV evidence from Notebooks 16, 17, and 22. The v1 analysis script checks the shared-start primary comparison. The v1 figure script requires complete v1 records and writes one A–D results dashboard plus a separate E architecture diagram. Neither v1 command trains a model or downloads data. See the [case study](CASE_STUDY.md) for captions, scope, and limits.
