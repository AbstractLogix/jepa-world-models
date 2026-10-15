# JEPA World Models Curriculum Guide

This guide defines a lecture path for graduate-level learners. Read the [coverage map](docs/CURRICULUM_COVERAGE.md) for notebook limits, the [evaluation protocol](docs/EVALUATION_PROTOCOL.md) before comparing results, the [reproduction matrix](docs/REPRODUCTION_MATRIX.md) for paper claims under study, and the [roadmap](docs/ROADMAP.md) for future lessons.

## Recommended Core Sequence

1. `notebooks/01_intuition.ipynb`
   - Outcome: Build intuition about collapse and JEPA design choices.
2. `notebooks/02_lejepa_from_scratch.ipynb`
   - Outcome: Run a small LeJEPA-inspired CIFAR-10 experiment and inspect representation diagnostics.
3. `notebooks/03_rectified_lpjepa.ipynb`
   - Outcome: Inspect a small Rectified LpJEPA-inspired loss and its sparsity trade-offs.
4. `notebooks/05_value_guided_planning.ipynb`
   - Outcome: Bridge latent representations to value-guided planning.
5. `notebooks/06_gradient_based_planning.ipynb`
   - Outcome: Compare random-shooting/CEM baselines with gradient-based planning.

Optional teaching analogues: `04`, `07`, `08`, `10`, `11`, `12`, `14`, `16`, and `17`. They do not reproduce the cited papers. Notebook `09` is an optional V-JEPA 2.1 checkpoint lesson. It uses 16 frames at 384×384 pixels and has a CPU path. Its feature check uses synthetic motion and does not reproduce a paper result. Notebook `10` compares planning horizon, feedback, and a scripted reference in continuous MountainCar; see its [local evidence report](docs/reproductions/planning-limits.md) and read its limits before comparing its results with the paper. Notebook `11` predicts real CIFAR-10 class-prompt embeddings and compares them with a linear classifier; it is supervised and closed-set. It needs the optional `vljepa` dependencies and about 300 MB of first-run downloads. See its [quickstart](specs/005-vljepa-real-labels/quickstart.md) and [local evidence report](docs/reproductions/vl-jepa.md). Notebook `12` compares continuous and discrete latent actions on generated clips with fixed image moments. It runs on CPU without data downloads; see its [local run report](docs/reproductions/latent-actions.md). Notebook `13` is an optional Oxford-IIIT Pet transfer study; see its [run report](docs/reproductions/object-centric-lejepa.md). Notebook `14` measures action interventions and paired policy outcomes in MountainCar; its [run report](docs/reproductions/decision-usefulness.md) shows why one-step accuracy does not ensure useful long-horizon planning. Notebook `15` checks symplectic area preservation with a harmonic-oscillator example and separates that structure from prediction error. It also reports seed-0 K=0 values and a separate K=1-fitted rollout curve for one local seed per model; this is partial evidence, not a paper reproduction. All notebook cells passed a fresh CPU-kernel run; see the [execution record](docs/evidence/lehami-symplectic-tutorial.md) and [full run record](docs/reproductions/lehami-actuated-velocity.md). Notebook `16` estimates position and velocity from rendered MountainCar frames, fits an action-conditioned model, and compares neutral-start CEM, controller-prior CEM, and direct control. It uses no JEPA weights or external data. The [local report](docs/reproductions/action-video-planning.md) records prediction, intervention, paired outcome, and compute results; they apply only to this simulator setup. Notebook `17` uses a polynomial state-based dynamics ensemble to compare random-action and planner-induced interval coverage, then tests mean-return CEM, uncertainty-penalized CEM, and direct momentum control. It is a CPU local analogue. See the [run report](docs/reproductions/model-uncertainty-planning.md) and read its claim limits.

Notebook `09` needs an opt-in download of about 1.66 GB. The CPU execution and its limits are recorded in [the local run report](docs/evidence/vjepa21-local-run.md).

## Timing Guidance (per 2-hour session)

- Session 1: 01 + intro context (90 min content + 30 min Q&A)
- Session 2: 02 walkthrough + precomputed checkpoint evaluation
- Session 3: 03 implementation extension
- Session 4: 05 planning bridge
- Session 5: 06 GRASP and comparison
- Session 6+: Advanced modules and capstones (04/07/08)
- Optional: 09 for dense video features (allow extra time for the checkpoint download and CPU load)
- Optional: 10 for planning horizons and feedback (CPU run, no download)
- Optional: 11 for text-embedding prediction (install the `vljepa` extra; allow time for about 300 MB of first-run downloads)
- Optional: 12 for generated-video latent actions (CPU run, no download)

## Prerequisites

- PyTorch basics (forward/backward, optimizers)
- Representation learning basics
- Intro RL concepts (state, action, return, value)

## Common Failure Modes

- OOM in notebook 02 → reduce batch size and/or use checkpoint-first mode
- Missing planning dependency → run `poetry install` for Gymnasium
- Noisy t-SNE interpretation → compare with linear probe + covariance stats

## Expected Core Outputs

- 01: clear visual contrast of collapse vs regularized embeddings
- 02: working checkpoint + linear-probe metric + t-SNE/embedding stats
- 05/06: report planning returns against a simple baseline; improvement is an outcome to measure, not a guarantee
- 10: separate local MountainCar measurements from the paper's Meta-World results
- 11: compare real class-prompt embedding prediction with a linear classifier; treat the small supervised result as an analogue only
- 14: compare held-out action interventions with closed-loop policy outcomes; keep the local task and compute limits explicit
- 16: connect rendered visual observations to action-conditioned prediction and planning; compare CEM priors with direct control and report held-out interventions, paired outcomes, and compute

## Lecturer Runbook (Week-by-Week)

### Before class

- Verify environment once:
  - `poetry install`
  - `poetry run pip install -r requirements-torch.txt`
- Open notebooks in order and run first import/setup cell in each.
- Precompute checkpoints for faster live delivery:
  - `notebooks/checkpoints/lejepa_cifar.pt`
  - `notebooks/checkpoints/rectified_lpjepa_cifar.pt`
  - `notebooks/checkpoints/value_guided_cartpole.pt`

### During class

- Use "checkpoint-first" flow for live reliability:
  1. Show model/loss definitions.
  2. Load checkpoint.
  3. Focus on diagnostics, plots, and interpretation.
- Keep one backup path ready:
  - If training slows, skip to evaluation cell with saved weights.
- Ask one conceptual question per section before running code.

### After class

- Export key plots (embedding clouds, reward curves) for review slides.
- Assign one hyperparameter sweep as a reproducibility task.

## Assessment Rubric (100 points)

- **Conceptual understanding (30 pts)**
  - JEPA vs generative world model tradeoffs
  - Collapse mechanisms and isotropy motivation
- **Implementation quality (30 pts)**
  - Clean notebook execution from top to bottom
  - Correct use of encoder/predictor/loss components
- **Evaluation rigor (25 pts)**
  - Reports linear probe + embedding geometry metrics
  - Compares at least one planning baseline
- **Analysis and communication (15 pts)**
  - Interprets failures, proposes fixes, and justifies design choices

## Suggested Assignments

1. **Assignment A (Core):**
    Rerun the notebook 02 baseline with one architecture change (for example, width or depth) and record the conditions.
2. **Assignment B (Planning):**
    Tune horizon/candidates in notebook 05 and report compute-vs-return tradeoff.
3. **Assignment C (Advanced):**
    Extend notebook 07 or 08 with one additional ablation factor.

## Troubleshooting Cheatsheet

- **`tqdm` widget warning in notebooks**
  - Ensure notebook kernel points to project `.venv`.
  - Reinstall widget stack in that env:
    - `poetry install`
- **GPU unsupported warning**
  - Use CUDA 12.8 wheel index in `requirements-torch.txt` (already configured).
- **Physical robotics setup**
  - Notebooks 07/08 use Gymnasium CartPole, and notebook 10 uses continuous MountainCar. They do not need MuJoCo and do not validate a robot.
