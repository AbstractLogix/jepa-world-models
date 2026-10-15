# JEPA World Models Tutorial

**From predictive representations to action selection.** This course uses runnable notebooks to teach JEPA ideas, explore recent research, and measure what a model can and cannot do.

[Start learning](CURRICULUM_GUIDE.md) · [Browse lessons](docs/CURRICULUM_COVERAGE.md) · [Read the research case study](docs/portfolio/CASE_STUDY.md) · [Explore the research roadmap](docs/ROADMAP.md) · [Read the reproduction matrix](docs/REPRODUCTION_MATRIX.md)

This repository has 21 notebooks. It separates paper-reported results from local measurements and teaching analogies. A toy result is never presented as a paper reproduction.

## Learning path

The core path moves from representation learning to planning, then asks whether those predictions help a policy choose actions.

```mermaid
flowchart LR
    A["01 · Intuition"] --> B["02–03 · Representation learning"]
    B --> C["05–06 · Latent planning"]
    C --> D["14, 16, 17, 20, 21 · Decision evaluation"]
    E["Optional research lessons · 09–21"] -. "extend the path" .-> B
    E -.-> C
```

| Step | Notebook | What you will learn |
| --- | --- | --- |
| 1 | [01 · Intuition](notebooks/01_intuition.ipynb) | See how predictive learning works and how collapse can occur. |
| 2 | [02 · LeJEPA](notebooks/02_lejepa_from_scratch.ipynb) and [03 · Rectified LpJEPA](notebooks/03_rectified_lpjepa.ipynb) | Train small models and compare representation objectives. |
| 3 | [05 · Value-guided planning](notebooks/05_value_guided_planning.ipynb) and [06 · Gradient-based planning](notebooks/06_gradient_based_planning.ipynb) | Use learned representations to plan actions in a small control task. |
| 4 | [14 · Decision usefulness](notebooks/14_decision_usefulness.ipynb), [16 · Action-conditioned planning](notebooks/16_action_conditioned_video_planning.ipynb), [17 · Uncertainty](notebooks/17_model_uncertainty_planning.ipynb), [20 · Policy-noise planning](notebooks/20_policy_noise_planning.ipynb), and [21 · Sampling-based planning](notebooks/21_sampling_based_planning.ipynb) | Measure decision outcomes and compare ways to search action sequences. |

The [curriculum guide](CURRICULUM_GUIDE.md) gives a teaching sequence. The [coverage map](docs/CURRICULUM_COVERAGE.md) lists every notebook, its evidence, and its limits.

## A result worth studying

In Notebook 14's MountainCar study, learned-model MPC reached the goal on 0 of 20 starts. A direct policy reached it on all 20. This is a local state-based result, not a JEPA or paper result. It shows why a good prediction score alone does not prove that a model helps with decisions. Read the [full report](docs/reproductions/decision-usefulness.md).

## Install and run

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv pip install -r requirements-torch.txt  # CUDA 12.8; omit for CPU-only use
uv run jupyter lab
```

The lock uses CPU PyTorch wheels by default. On a CUDA 12.8 system, install the matching wheels after `uv sync --locked`. Open `notebooks/01_intuition.ipynb` first. Notebook 11 needs optional text-embedding dependencies. Select the extra before installing or replacing PyTorch:

```bash
uv sync --locked --extra vljepa
uv pip install -r requirements-torch.txt  # optional CUDA 12.8 override
```

For CPU-only setup, use the locked CPU wheels and omit the CUDA install command. `uv sync --locked` restores those CPU wheels; on a GPU host, reapply the CUDA override after each sync.

This project uses `uv` for Python environments and installs. If the system drive is low on space, set `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, and `JEPA_ASSETS_DIR` to folders on E: or D: before setup. Set `HF_HUB_CACHE`, `HF_ASSETS_CACHE`, and `TORCH_HOME` too if model downloads must stay on that drive. In WSL, for example:

```bash
export UV_CACHE_DIR=/mnt/e/codex-research/uv-cache
export UV_PROJECT_ENVIRONMENT=/mnt/e/codex-research/jepa-world-models-tutorial/.venv
export JEPA_ASSETS_DIR=/mnt/e/codex-research/jepa-world-models-assets
export HF_HUB_CACHE=/mnt/e/codex-research/huggingface/hub
export HF_ASSETS_CACHE=/mnt/e/codex-research/huggingface/assets
export TORCH_HOME=/mnt/e/codex-research/torch
mkdir -p "$JEPA_ASSETS_DIR" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$TORCH_HOME"
uv sync --locked
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -r requirements-torch.txt
```

Replace `/mnt/e` with `/mnt/d` to use D:. For Windows PowerShell commands and path checks, see the [external-storage quickstart](specs/024-external-asset-storage/quickstart.md). The quickstart does not move existing files. Check the [coverage map](docs/CURRICULUM_COVERAGE.md) before a lesson that downloads assets.

Some optional lessons download data or model weights. Notebook 09 uses an approximately 1.66 GB checkpoint, Notebook 11 uses about 300 MB of first-run assets, and Notebook 13 uses about 800 MB of dataset files. Check the [coverage map](docs/CURRICULUM_COVERAGE.md) before starting a lesson.

## Evidence and research

Each result has an explicit status:

- **Paper-reported** — stated by a cited source.
- **Locally measured** — produced by this repository under recorded conditions.
- **Illustrative** — a teaching example, not a paper result.
- **Not run** — no local execution evidence is available.

Use the [evaluation protocol](docs/EVALUATION_PROTOCOL.md) before making a comparison. The [research case study](docs/portfolio/CASE_STUDY.md) summarizes the strongest local findings and their limits. The [research roadmap](docs/ROADMAP.md) and [reproduction matrix](docs/REPRODUCTION_MATRIX.md) track current sources, planned lessons, resource needs, and open reproduction work. The full [documentation index](docs/index.md) links to reports and evidence records.

## Contribute

Contributions that improve a lesson, add a careful evaluation, or make the course easier to follow are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md). For a new lesson or research reproduction, use the project's [Spec Kit quickstart](specs/001-project-vision/quickstart.md).
