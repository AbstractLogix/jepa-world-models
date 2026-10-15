# JEPA World Models – Educational Repo (2026 edition)

Hands-on lessons on **Joint Embedding Predictive Architecture (JEPA)**, from representation learning to simple latent planning. Notebook 09 adds optional V-JEPA 2.1 video-feature inspection, notebook 10 compares planning limits in a small control task, notebook 12 compares latent actions from generated clips, notebook 13 studies mask-pooled object features, notebook 14 measures action interventions and paired policy outcomes, notebook 15 explains symplectic latent dynamics, notebook 16 connects rendered video to action-conditioned planning, notebook 17 measures uncertainty calibration and model-selected plan gaps, and notebook 19 teaches H-JEPA's top-down hierarchy with a CPU analogy. The roadmap includes pretrained-model tutorials and released-checkpoint evaluations; the JEPA-WAM LIBERO-Plus re-evaluation remains pending.

The notebooks use small teaching examples. None reproduces a full cited paper result or validates a physical robot. Notebook 11 is a supervised class-prompt analogue. Notebook 12 is a synthetic video analogue. Notebook 13 is an exploratory Oxford-IIIT Pet transfer study. Notebook 14 is a MountainCar decision-usefulness analogue, not a JEPA result. Notebook 15 is a mathematical analogue with partial paper-linked evidence; its local pendulum result uses one seed per model. Notebook 16 is a rendered MountainCar analogue, not a JEPA implementation or a reproduction of V-JEPA 2-AC or PiJEPA. Notebook 17 is a state-based MountainCar analogue, not a JEPA model or a paper reproduction. Notebook 18 teaches Qantara's multi-paradigm inference paths with a CPU analogy; no Qantara assets or local paper results are available. Notebook 19 teaches H-JEPA's top-down planning flow with hand-written toy summaries; it does not reproduce the Visual AntMaze result. See the [curriculum coverage map](docs/CURRICULUM_COVERAGE.md), [evaluation protocol](docs/EVALUATION_PROTOCOL.md), [paper-reproduction matrix](docs/REPRODUCTION_MATRIX.md), and [research roadmap](docs/ROADMAP.md) for current limits, reporting rules, and planned lessons.

## Core Concepts

- How JEPA and generative models differ in their prediction objectives and trade-offs
- Theoretical foundations: isotropic Gaussians, sparsity, maximum-entropy representations
- Scaling to video, language, actions, planning in latent space
- Value shaping, gradient-based planning, latent actions, and reproducible evaluation

Research references and priorities are listed in the [roadmap](docs/ROADMAP.md). Start with LeCun's 2022 position paper, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf), then follow the sources linked there.

## Notebooks

Core path: `01` → `02` → `03` → `05` → `06`. Optional teaching analogues: `04`, `07`, `08`, `10`, `11`, `12`, `14`, `16`, and `17`. Optional V-JEPA 2.1 checkpoint lesson: `09`. Optional mask-based transfer study: `13`.

1. `01_intuition.ipynb` – JEPA concepts with a synthetic 2D example
2. `02_lejepa_from_scratch.ipynb` – Small LeJEPA-inspired CIFAR-10 experiment with simplified SIGReg
3. `03_rectified_lpjepa.ipynb` – Small CIFAR-10 experiment with Rectified LpJEPA-inspired losses
4. `04_vjepa_plus_v1_jepa.ipynb` – Vision-language toy bridge with synthetic text targets
5. `05_value_guided_planning.ipynb` – Value-guided latent planning in CartPole
6. `06_gradient_based_planning.ipynb` – Random shooting and gradient-planning comparison in CartPole
7. `07_latent_action_world_models.ipynb` – Discrete latent-action toy model in CartPole
8. `08_physical_planning_ablation.ipynb` – CartPole ablations inspired by physical-planning research
9. `09_vjepa21_dense_features.ipynb` – Inspect dense features from an official V-JEPA 2.1 checkpoint; its synthetic correspondence example is not a paper reproduction
10. `10_planning_limits.ipynb` – Compare imagined planning, feedback MPC, and scripted reference subgoals in MountainCar; this is not a Meta-World reproduction
11. `11_vl_jepa.ipynb` – Predict real CIFAR-10 class-prompt embeddings and compare with a linear classifier; this is a supervised analogue, not a VL-JEPA reproduction
12. `12_video_latent_actions.ipynb` – Compare bounded continuous actions with discrete codes on generated clips; this is a synthetic analogue, not a paper reproduction
13. `13_object_centric_lejepa.ipynb` – Compare image-level and mask-pooled objectives with Oxford-IIIT Pet trimaps; the completed local transfer study is not the COCO paper reproduction
14. `14_decision_usefulness.ipynb` – Measure held-out action interventions and compare learned-model MPC with a direct policy on paired MountainCar starts; this is a local analogue, not a JEPA result
15. `15_lehami_symplectic_dynamics.ipynb` – Check area preservation and trajectory error in a harmonic-oscillator example; compare the paper's K=0 probe values and a separate K=1-fitted rollout curve with one local seed per model, without calling either a reproduction
16. `16_action_conditioned_video_planning.ipynb` – Fit a small action-conditioned model from rendered MountainCar frames and compare two CEM starts with a direct controller; this is a simulator analogue, not a JEPA result
17. `17_model_uncertainty_planning.ipynb` – Calibrate a small state-based ensemble under random and planner actions, then compare selected-plan predictions and outcomes; this is a local analogue, not a JEPA or paper reproduction
18. `18_qantara_three_inference_modes.ipynb` – Explain Qantara's planning, behaviour-cloning, and video–inverse paths with a CPU analogy; this is not a Qantara evaluation
19. `19_hjepa_hierarchical_planning.ipynb` – Trace top-down H-JEPA subgoals with hand-written state summaries; this is not a learned hierarchy or paper reproduction

Notebook 09 uses the official ViT-B/384 checkpoint (about 1.66 GB) with a 16-frame 384×384 input. It has a CPU path. Its synthetic clip check does not reproduce a paper result.

Notebook 11 needs the optional `vljepa` dependencies and about 300 MB of first-run data and model downloads. See its [quickstart](specs/005-vljepa-real-labels/quickstart.md) for data, checkpoint, and license limits.

Notebook 12 runs on CPU without downloads. It uses generated clips and fixed image moments. See the [local run report](docs/reproductions/latent-actions.md) for settings and limits.

Notebook 13 downloads about 800 MB of Oxford-IIIT Pet images and annotations. Its five-seed local transfer study is complete; see the [run report](docs/reproductions/object-centric-lejepa.md) for its scope and limits.

Notebook 14 runs on CPU without downloads. It records 60 action interventions and 20 paired policy starts. See the [local evidence report](docs/reproductions/decision-usefulness.md); its MountainCar result does not establish a general model advantage.

Notebook 15 uses Python's standard library and needs no downloads. Its oscillator calculation is a mathematical analogue. It displays the paper's three-seed results separately from local seed-0 K=0 and K=1-fitted rollout measurements; neither local result is a full reproduction. All notebook cells passed a fresh CPU-kernel run; see the [execution record](docs/evidence/lehami-symplectic-tutorial.md) and the [full pendulum run report](docs/reproductions/lehami-actuated-velocity.md).

Notebook 16 uses rendered MountainCar frames and the repository's existing NumPy, Gymnasium, Pillow, scikit-learn, and Matplotlib dependencies. It needs no checkpoint or dataset download. Its local run found 0/20 successes for neutral-start CEM and 20/20 for both controller-prior CEM and the direct controller; the direct controller reached the goal sooner on average. This result applies only to the stated setup. See the [run report](docs/reproductions/action-video-planning.md) and its [episode](docs/evidence/action-video-planning-episodes.csv) and [intervention](docs/evidence/action-video-planning-interventions.csv) records.

Notebook 17 uses CPU-only MountainCarContinuous state data and existing project dependencies. It needs no checkpoint or dataset download. The [quickstart](specs/018-model-uncertainty-planning/quickstart.md) explains how to run it. The [run report](docs/reproductions/model-uncertainty-planning.md) records five model runs, calibrated prediction intervals, paired policy outcomes, and selected-plan gaps. It is a local state-based analogue, not a JEPA model or paper reproduction.

Notebook 18 runs on CPU without external assets. Its Qantara three-path example is a deterministic analogy; the released-checkpoint evaluation remains gated by data provenance and renderer checks. See the [execution record](docs/evidence/qantara-inference-tutorial.md) and [preflight](docs/reproductions/qantara-preflight.md).

Notebook 19 runs on CPU with the standard library. Its state summaries and top-down subgoal rules are hand-written; the separate Visual AntMaze checkpoint evaluation remains blocked before a task starts. See the [execution record](docs/evidence/hjepa-hierarchical-planning.md) and [preflight](docs/reproductions/h-jepa-antmaze-preflight.md).

## Folders

```bash
notebooks/                # main tutorials
src/jepa_tutorial/        # losses, evaluation helpers, and value shaping
src/planning/             # CartPole planning examples
src/viz/                  # embedding, value-shaping, and rollout plots
configs/                  # experiment settings
docs/                     # coverage, evaluation, reproduction, and research roadmap
specs/                    # project and feature specifications
```

## Quick Start

Requires **Python 3.10–3.13** and [Poetry](https://python-poetry.org/docs/#installation).

```bash
# From the repository root

# Step 1 – install all PyPI deps (numpy, scipy, gymnasium, timm, jupyterlab …)
poetry install

# Optional notebook 11 text-embedding dependencies
poetry install --extras vljepa

# Step 2 – install PyTorch + torchvision + torchcodec from the PyTorch wheel index
#           (bundles its own CUDA libs – no system CUDA install needed)
poetry run pip install -r requirements-torch.txt

# CPU-only machines (no GPU): use the cpu index instead
# poetry run pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu

# Step 3 – launch notebooks
poetry run jupyter lab
```

Notebooks 05–08 use Gymnasium CartPole. Notebooks 10 and 14 use Gymnasium continuous MountainCar. These lessons do not need `dm_control` or test physical robot control. Notebook 11 needs a first-run network connection unless its data and models are cached. Start with `notebooks/01_intuition.ipynb`.

## References / Starters

- [V-JEPA 2 code and checkpoints](https://github.com/facebookresearch/vjepa2)
- Research sources and planned lessons: [docs/ROADMAP.md](docs/ROADMAP.md)
- Current lesson scope and limits: [docs/CURRICULUM_COVERAGE.md](docs/CURRICULUM_COVERAGE.md)
- Local paper-reproduction attempts: [docs/REPRODUCTION_MATRIX.md](docs/REPRODUCTION_MATRIX.md)
- New work uses the project specifications in `specs/` and the Codex Spec Kit skills in `.agents/skills/`.

For a new change, use `$speckit-specify`, `$speckit-plan`, and `$speckit-tasks` before implementation. Review alignment with `$speckit-analyze`, then check remaining work with `$speckit-converge`. See [the contributor quickstart](specs/001-project-vision/quickstart.md).

PRs encouraged: implementations of missing papers, better visualizations of value-distance shaping, or integrations with real robot data.

**Current tech stack**

- PyTorch
- torchvision (image data)
- matplotlib + scikit-learn (plots and probes)
- gymnasium (CartPole and MountainCar planning examples)

Notebook 09 needs an optional 1.66 GB checkpoint download. Video training and robot data remain planned extensions. See the roadmap for their access and compute limits.
