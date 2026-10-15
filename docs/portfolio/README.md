# Portfolio files

This folder contains the public case study and a JSON manifest for the Abstract Software website.

## Website use

`project.json` is plain JSON. A SvelteKit site can import it as a build-time data module or copy it into the site's project-data folder. Keep the `schemaVersion` field and render only fields the page understands. The manifest uses repository-relative paths for local evidence and media. A site build should copy those files into its static assets or resolve them against the repository URL at a pinned commit.

Example:

```ts
import project from "$lib/data/projects/jepa-world-models.json";
```

Use `media[].alt` as the image alternative text and `media[].caption` as the visible caption. Show each finding's `status`, `scope`, and `claimBoundary` with the summary. Do not present the MountainCar analogues as JEPA results or paper reproductions.

## Claim checks

| Claim | Source to inspect | Safe wording |
| --- | --- | --- |
| SIGReg probe difference | [Notebook 02 report](../reproductions/lejepa-rectified-lpjepa.md) | Small five-seed CIFAR-10 objective study; interval includes zero. |
| RDMReg projector result | [Notebook 03 report](../reproductions/lejepa-rectified-lpjepa.md) | Local projector-probe increase over a weak teaching proxy; not ImageNet-100. |
| Decision-usefulness contrast | [Notebook 14 report](../reproductions/decision-usefulness.md) and saved notebook outputs | State-model MPC failed on these 20 starts; direct momentum rule succeeded; actual compute differed. |
| Prior-guided search analogue | [Notebook 16 report](../reproductions/action-video-planning.md) and [episode CSV](../evidence/action-video-planning-episodes.csv) | A hand-written prior improved this local CEM setup; it is not a learned policy or JEPA result. |
| Calibration under planner shift | [Notebook 17 report](../reproductions/model-uncertainty-planning.md) and [raw rows](../evidence/model-uncertainty-calibration.csv) | Five state-model fits had wider planner-shift uncertainty; one task only. |
| Official V-JEPA checkpoint execution | [Notebook 09 run record](../evidence/vjepa21-local-run.md) | Checkpoint inference on one synthetic clip; no benchmark or planning claim. |

“Verified” in the manifest means the value was checked against local records. It does not mean an independent party repeated the experiment. The current portfolio branch is stacked on draft PR #1. Wait for maintainer review and merge before treating these files as part of the default repository branch.

## Rebuild and validate

From the repository root, on the current `uv` branch:

```bash
uv sync --locked
uv run --locked --no-sync python scripts/generate_portfolio_figures.py
uv run --locked --no-sync python -m unittest discover -s tests
```

The figure script reads the saved Notebook 14 episode table and the retained Notebook 16 and 17 CSV evidence. It does not train a model or download a dataset. See the [case study](CASE_STUDY.md) for figure captions, scope, and limits.
