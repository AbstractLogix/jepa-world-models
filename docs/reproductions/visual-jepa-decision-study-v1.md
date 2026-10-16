# Matched visual JEPA decision study v1

**Status**: Complete 660-episode matrix; partial scientific comparison because all five JEPA seeds triggered the low-spread rule. JEPA planner outcomes remain exploratory.<br>
**Task**: Synthetic `MountainCarContinuous-v0`, goal position 0.45, 300-step episode cap.<br>
**Protocol**: [Frozen preregistration](protocols/visual-jepa-decision-study-v1.md)<br>
**Lesson**: [Notebook 23](../../notebooks/23_visual_jepa_decision_study.ipynb)

## Question

Does an action-conditioned visual JEPA improve decisions beyond a direct visual-momentum controller and beyond the effect of planner initialization?

This is a local, small-scale study. It is not a reproduction of V-JEPA 2, LeJEPA, VISReg, or another paper. Recent evaluation work motivates the focus on interventions, closed-loop outcomes, ranking, uncertainty, and cost, but its findings are not local results. See [Yu et al.](https://arxiv.org/abs/2606.15032), [Jena et al.](https://arxiv.org/abs/2609.29669), [V-JEPA 2](https://arxiv.org/abs/2506.09985), and the [physical-planning JEPA-WM study](https://arxiv.org/abs/2512.24497).

## Method

The study compares exact dynamics, learned state-space dynamics, learned visual reconstruction dynamics, action-conditioned visual JEPA dynamics, and a direct visual-momentum controller. The direct controller is hand-written. It is not a learned policy. The state-space model and exact dynamics receive privileged state input. Visual models receive generated frames; a frozen linear state readout scores their planner rollouts.

Five shared random-trajectory groups provide 4,800 training rows and 1,200 validation rows each. Five model seeds train each of the three model families for 35 epochs. Separate representation-test, intervention, planner-development, confirmation, and final-start seeds are fixed in the preregistration. The final episode matrix shares 20 evaluation starts across methods and model seeds.

Planner settings were selected on development starts before final evaluation. The selected exact-dynamics planner passed the development and confirmation gate at 20/20, then passed the held-out oracle gate at 20/20. The selected setting uses a 60-step horizon, 48 candidates, 8 elites, 4 updates, and replanning every five environment steps. The rollout score is cumulative Gymnasium return.

The protocol names one initialization “uniform.” In the implementation it means no momentum prior: action noise is sampled from a zero-mean, unit-standard-deviation Gaussian and clipped to `[-1, 1]`. It is not a mathematical uniform distribution.

## Training and representation diagnostics

All 15 planned model fits completed on an NVIDIA RTX 5070. The E:-hosted runtime used PyTorch 2.10.0+cu128 and CUDA 12.8. Peak allocated GPU memory was below 600 MiB per model fit. The five action-conditioned JEPA runs all triggered the implementation's low-spread flag: all 32 latent dimensions had standard deviation below `1e-3` on representation-test trajectories.

The same JEPA models retained strong position-probe scores (R² 0.973–0.996) and weak velocity-probe scores (R² −0.012–0.095). This is evidence of a low-spread representation with uneven decoded state information. It is not proof that every latent is uninformative. The preregistration says a collapsed model makes the comparison partial, but it does not repeat the numerical threshold. Treat that gap as a protocol limitation.

The learned objective is a local action-conditioned predictive loss with a LeJEPA-style SIGReg term. It is not the published LeJEPA or V-JEPA 2 objective as a whole. No external dataset or pretrained checkpoint was used.

## Current evidence status

- Development search: 20/20 starts passed; selected configuration was locked before final evaluation.
- Confirmation set: 10/10 starts passed.
- Held-out exact-dynamics gate: 20/20 starts passed.
- Model fits: 15/15 complete; five JEPA models triggered the low-spread diagnostic.
- Held-out interventions: 960/960 rows are retained.
- Final episode matrix: 660/660 complete rows passed the evidence validator. The run used 408,064,200 of the 750,000,000 transition-call ceiling.

The matrix and validation checks passed. The JEPA comparison remains partial under the preregistered low-spread rule. Any outcome from these flagged models is an exploratory diagnostic, not evidence of a valid JEPA planning advantage.

## Episode outcomes

| Method | No momentum prior | Momentum prior | Access and evidence limit |
| --- | ---: | ---: | --- |
| Exact-dynamics CEM | 0/20 | 20/20 | Privileged state; the prior setting passed the held-out oracle gate. |
| Learned state-space CEM | 0/100 | 99/100 | Privileged state; five paired model/data seed groups. |
| Image-reconstruction CEM | 2/100 | 17/100 | Generated frames; five paired model/data seed groups. |
| Action-conditioned JEPA CEM | 0/100 | 0/100 | Generated frames; all five JEPA seeds triggered low spread. |
| Direct visual-momentum controller | — | 20/20 | Handcrafted controller; zero model calls and 0.11 seconds mean episode time. |

On the common held-out intervention score, the state model ranked 128/225 non-tied action pairs correctly (56.9%), the image model 116/225 (51.6%), and the JEPA 131/225 (58.2%). JEPA's mean position-probe R² was 0.973–0.996, while velocity-probe R² was −0.012–0.095. These are diagnostics, not a substitute for closed-loop success.

The supplementary paired-start analysis estimated a mean return difference of −89.356 for momentum-prior JEPA CEM minus direct control. Its 95% hierarchical bootstrap interval was [−89.428, −89.279]. This is descriptive and is not a significance test. The low-spread representation, shared oracle-gated starts, five paired model/data seeds, and one synthetic task limit the estimate.

## Methodology audit

The train, validation, probe-test, intervention, development, confirmation, and final-start roles use separate declared seeds. Each of the five model seeds is paired with one training-data seed. This protects the held-out starts, but the design cannot separate model-initialization variation from training-data variation. Treat each result as a paired model/data seed, not as an independent model-only replicate.

The exact simulator and learned state-space model use privileged state. The image-reconstruction and action-conditioned JEPA methods use generated frames. Their linear position/velocity readouts are fit from training data and selected with validation data. The direct visual-momentum controller is handcrafted and uses color-centre features. It is one fixed-controller outcome on each of the 20 shared starts; it is not repeated for each model seed.

Notebook 22 used a VISReg-style temporal-prediction adaptation. Study v1 uses action-conditioned latent prediction with a LeJEPA-style SIGReg regularizer. These are separate local objectives. Neither is the complete published objective or a paper reproduction.

The new CEM settings were selected on development starts and checked on confirmation starts before the final oracle gate. This makes the v1 exact-dynamics check a valid gate for this task and protocol. Notebook 22's earlier 0/20 oracle outcome remains a historical result under its own setup; the new gate does not change or overwrite it.

The same 20 final starts first serve as the predeclared exact-oracle gate and then as paired starts for the comparison. The gate does not tune model parameters, but it determines whether the comparison opens. Read all learned-method outcomes as conditional on this gate passing, not as an unconditional sample of task difficulty.

The frozen matrix does not include a prior-only CEM arm. Direct control shows what the handcrafted momentum rule can do without a planner; uniform and momentum-prior CEM compare search initialization while using learned dynamics. These arms do not measure prior-only search. This gap limits conclusions about how much of a planner's outcome comes from its learned transition model.

The run's original `report.json` bootstrap implementation resamples evaluation starts separately within each selected training seed. The starts are shared across seeds, so that procedure does not preserve the planned paired-start sampling. The supplementary analysis command uses one shared resampled start set per bootstrap replicate and retains the original report as execution evidence. The interval is descriptive; it is not a significance test.

The v1 study does not estimate calibrated model uncertainty. It records a fixed embedding-spread warning, probe diagnostics, interventions, and outcomes. Transition-model calls are a useful workload count, not an equal-compute measure across state, image, and latent models. The exact oracle ran on CPU; learned methods ran on GPU.

Learned CEM methods averaged 0.46–0.69 million transition calls and 11.35–24.55 seconds per episode, depending on model and initialization. Exact-oracle CEM used the CPU and averaged 0.23–0.58 seconds; learned models used the GPU. The handcrafted direct controller used no model calls and averaged 0.11 seconds. These timings are not compute-matched. The original `report.json` remains the run record; the supplementary analysis JSON uses the paired-start bootstrap described above.

## Limits and claim boundary

This task uses generated images from one small simulator. The direct controller is handcrafted. The visual reward readout is an auxiliary evaluator. The intervention continuation uses simulator state and the same fixed momentum rule for every method; it is not a deployable learned reward model. Planner transition calls and elapsed time are reported separately and do not imply equal hardware work.

The study can show how one local planner, one synthetic task, and these compact models behave. It cannot establish robotics capability, a general JEPA result, a learned-policy comparison, statistical significance, or independent replication. The main future experiment remains a matched learned-JEPA versus learned-direct-policy evaluation on a named visual-control benchmark.

## Reproduction

Use `uv sync --locked` for the CPU environment. The CPU smoke and test path is:

```bash
uv run --locked --no-sync python -m unittest discover -s tests
uv run --locked --no-sync python scripts/run_visual_jepa_decision_study.py smoke --device cpu
```

For the full study, use the E:-hosted CUDA runtime and a separate output folder on E: or D:. Follow the ordered `dev-search`, `oracle-gate`, then `full` commands in the [feature quickstart](../../specs/029-matched-visual-jepa-study/quickstart.md). The full run uses fixed seeds and a 750-million transition-call ceiling. It does not download assets.

The raw development, oracle, model, probe, intervention, episode, and paired-analysis files are retained under `docs/evidence/`. The E: volume holds model checkpoints and working outputs. The evidence validator and two deterministic figure commands are listed in the quickstart.

## Validation record

| Check | Result |
| --- | --- |
| Unit tests | Passed: 46 tests. |
| CPU smoke | Passed: 1,800 transitions, all three model families, and four planner episodes; final evaluation cases remained closed. |
| CUDA study | Completed separately on the RTX 5070. The complete 660-episode matrix passed evidence validation; the run used 408,064,200 of 750,000,000 transition calls. |
| Evidence validator | Passed: 200 development rows, 15 model runs, 10 probe rows, 960 intervention rows, and 660 episode rows. |
| Ruff | Passed for `src`, `tests`, and `scripts`. Two existing Ruff I001 findings were corrected. |
| Package build | Passed: source archive and wheel built to the E: drive. |
| Notebook | Passed: notebook schema and Python code-cell checks. Stable IDs were added to all nine cells. |
| Figures | Passed: PNG dashboard, accessible SVG architecture, and metadata regenerated byte-for-byte in a second E: output folder. |
| Portfolio manifest and local links | Passed: JSON parsed; local links in the changed public and feature documents resolved. |
| Spec Kit analysis and convergence | Passed: no uncovered feature requirements; no convergence tasks were appended. |
| Diff formatting | Passed: `git diff --check`. |
| Full JEPA-WAM LIBERO-Plus and 450-episode Push-T evaluations | Not run. |

The low-spread cutoff used by the implementation was not stated as a numeric threshold in the frozen preregistration. The report keeps this as a protocol limitation; the v1 protocol was not edited after results were known. Future studies must set this rule before model training.
