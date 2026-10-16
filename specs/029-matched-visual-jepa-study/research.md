# Research and Methodology Audit

**Reviewed**: 2026-10-16
**Scope**: Notebook 22, its visual dynamics models and VISReg-style objective, the planning and evidence path, the shared evaluation protocol, and the current reproduction matrix.

## Historical evidence to preserve

Notebook 22 is merged into `main`. Its raw model, intervention, pilot, and episode records remain the historical baseline. Its reported decision results are: 153/295 non-tied action pairs ranked correctly (51.9%); uniform-start JEPA CEM 0/100; momentum-prior JEPA CEM 100/100; direct visual-momentum control 20/20; exact-dynamics CEM 0/20. These outcomes apply to one synthetic MountainCar task and do not establish a general JEPA advantage.

The final Notebook 22 run used five training-data seeds (5100–5104), five model seeds (7221, 7227, 7239, 7253, 7263), 20 shared episode starts (3100–3119), and 20 held-out intervention states from seeds 2100–2104. Each training-data group used 16 episodes for training and 4 for validation. The intervention and episode seeds were separate from training. Direct control had one row per evaluation start; repeated pairing of those values with five model seeds was explicitly descriptive, not five independent direct-control samples.

## Methodology findings

| Area | Observed fact | Risk and response |
|---|---|---|
| Horizon | CEM planned 10 steps, while each episode allowed 300 steps. | MountainCar needs accumulated momentum over a much longer path. A short rollout can hide useful long-range action effects. Tune the oracle planner on development starts first. |
| Search budget | 24 candidates, 4 elites, 2 updates; CEM replanned at every environment step. It evaluated 480 candidate transitions per decision and up to 144,000 per 300-step episode. | Exact dynamics do not fix a poor horizon or an underpowered optimizer. Record calls and wall time. |
| Initial distribution | Uniform CEM used mean 0 and standard deviation 1. The momentum-prior CEM used a constant mean equal to the current hand-written controller action and standard deviation 0.5. | The prior condition changes initialization and can exploit the same rule used by direct control. Compare uniform and prior initialization within each model and retain direct control as a separate baseline. |
| Objective | Oracle/state methods scored only terminal position and velocity. The image model scored decoded pixel distance to a goal frame. JEPA scored latent distance to a goal embedding. The actual environment gives a per-step action cost and a terminal success reward. | These objectives have different units and can select different actions. New model-based planners use a common task-grounded state probe and task reward. Keep native losses as diagnostics only. |
| Action ranking | The 51.9% result aggregates within-model ranks using method-specific predicted and observed scores. | It is not a common task-outcome ranking. New interventions rank a common continuation outcome and report ties and denominators. |
| Prediction horizon | Models were trained with one-step prediction; CEM recursively used predictions for up to 10 steps. | One-step loss does not measure compounding error. Retain multi-step intervention fidelity and selected-plan prediction gaps. |
| Exact dynamics | Notebook 22 duplicates the Gymnasium transition equations in a Torch helper. It does not test parity against the installed environment. | Add deterministic transition parity checks before trusting the oracle gate. Gymnasium documents a force scale of 0.0015, gravity term 0.0025, clipped speed/position, per-step action cost, and +100 goal reward. |
| Visual observation | The deterministic renderer encodes position as a red blob and the goal as a green blob. The direct controller uses color-centroid position and two-frame displacement as velocity. | This is a deliberately simple synthetic observation. It does not test camera variation, occlusion, or robot perception. Label the controller as handcrafted, not learned. |
| Training objective | The action-conditioned model has a compact learned encoder and predictor; the local VISReg-style loss adapts a multi-view regularizer to temporal prediction. The model uses no published V-JEPA 2 checkpoint or robot data. | Keep it labeled as a local JEPA-style analogue and keep the adaptation distinct from VISReg, LeJEPA, and V-JEPA 2 paper results. |
| Statistical unit | Learned methods have five model seeds and 20 shared evaluation starts. Direct control has only 20 outcomes. | Model-seed uncertainty and evaluation-start variation are different sources. Do not duplicate direct outcomes and treat them as independent observations. Use paired, seed-aware descriptive intervals and report each seed. |
| Test separation | Notebook 22 separates training/validation trajectories, interventions, and evaluation starts. It has no planner-tuning development set. | The new study adds development-only planner tuning and a final oracle gate. Do not use final outcomes to tune the planner. |
| Oracle gate use | The same 20 final starts first determine whether the locked oracle passes, then serve as paired outcomes for all methods. | This does not tune learned models, but the comparison is conditional on the gate passing. State this selection rule when interpreting results. |
| Paired seed sources | Study v1 pairs each model seed with one training-data seed. | The five points vary both model initialization and sampled training data. Report them as model/data seed pairs; do not attribute their spread to model seeds alone. |
| Missing search control | Study v1 compares direct control with uniform and momentum-prior CEM over learned dynamics, but has no prior-only CEM condition. | The direct rule does not measure planner performance with the prior but without a learned transition model. Keep that comparison open. |
| Bootstrap implementation | The full runner's interval code independently resamples the shared evaluation starts within each selected model seed. | This does not preserve common-start resampling across seeds. Retain the emitted report for provenance and publish the corrected paired-start interval from the separate analysis script. |
| Uncertainty | Study v1 records a low-spread flag and probe/intervention diagnostics, but no calibrated predictive distribution. | Do not present it as a model-uncertainty or reliability-calibration experiment. Notebook 17 is a separate state-space analogue. |
| Compute | State and oracle methods use simulator state; visual methods use rendered frames. The oracle is timed on CPU and learned models on GPU. | Transition-call totals and elapsed time do not establish compute parity. Report access and device beside each result. |

## Source review

- Gymnasium's task definition documents the continuous action range, transition equations, reward, start distribution, and termination threshold: [MountainCarContinuous](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/).
- V-JEPA 2 trains an action-conditioned world model using a large pretrained video encoder and robot trajectories. This project's compact synthetic model is not a reproduction: [V-JEPA 2](https://arxiv.org/abs/2506.09985) and [official code](https://github.com/facebookresearch/vjepa2).
- The 2026 JEPA-WM design study varies model, objective, and planner choices; it fixes planning settings before subsequent model ablations. This supports the ordering of this study, not its expected result: [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/abs/2512.24497).
- CEM optimizes action sequences by sampling, scoring, and updating a distribution. Candidate count, horizon, and update count affect both search and compute: [Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization](https://proceedings.mlr.press/v120/bharadhwaj20a.html).
- The shared project protocol distinguishes prediction fidelity, interventions, closed-loop outcome, decision benefit, and compute: [Evaluation Protocol](../../docs/EVALUATION_PROTOCOL.md).
- A 2026 decision-focused evaluation position paper recommends separate evidence for counterfactual action fidelity, closed-loop rollouts, policy ranking, optimization lift, model exploitability, and uncertainty calibration: [Yu et al.](https://arxiv.org/abs/2606.15032).
- A survey reports that 11 of 160 collected benchmarks compare world models directly with policies, and four turn prediction into executed action: [Jena et al.](https://arxiv.org/abs/2609.29669). This motivates the matched-baseline roadmap; the survey does not establish a JEPA benefit.
- A recent control-grounded JEPA paper adds inverse dynamics and state alignment, then reports results on four robotic tasks: [Liu et al.](https://arxiv.org/abs/2609.03565). Its claims remain paper-reported until the release and setup are independently checked.
- A recent point-cloud JEPA study compares geometry and image observations and reports open code and weights: [Oberweger et al.](https://arxiv.org/abs/2608.29434). Check release terms before using its artifacts.
- PSG-JEPA adds training-only proprioceptive and multi-horizon joint-change targets to action-conditioned JEPA prediction. The authors report latent-probe, frozen-latent planning, and policy-learning results. Its public README lists one seed-3072 LIBERO checkpoint, while the paper reports three-seed results; the OGBench-Cube planning checkpoint is listed as upcoming. Review its code, dataset, simulator, and base-model terms before any run: [paper](https://arxiv.org/abs/2608.06799), [official repository](https://github.com/Haodong-Yan/PSG-JEPA).
- The October 2026 safety survey organizes open risks across data, observation, state and memory, dynamics, evaluation, planning, action, and feedback. It motivates a future safety lens but does not provide a single quantitative reproduction target: [Wang et al.](https://arxiv.org/abs/2610.13372).

## Decisions

1. Keep Notebook 22 and its CSV evidence unchanged. Put this follow-up in a new `v1` evidence namespace.
2. Use exact dynamics to tune a bounded CEM protocol on development starts. The predeclared search tests horizon, candidate count, iterations, initialization, and terminal versus cumulative task score.
3. Lock the planner before the final oracle gate. Do not open learned-model evaluation unless exact-dynamics planning succeeds on at least 18 of 20 final starts.
4. Compare oracle and state models as privileged controls. Compare learned image reconstruction and JEPA models from rendered inputs. Compare uniform and common visual-momentum-prior initialization within each planning model.
5. Train one shared, training-only linear state probe per visual model. Select its regularization on validation data and report position, velocity, and action-sensitivity diagnostics on separate test data. The probe is not evidence of planning value by itself.
6. Use episode success and Gymnasium return as task outcomes. Keep latent, pixel, and state prediction losses separate.
7. Run CPU checks and smoke independently. Use the prepared E:-hosted CUDA 12.8 environment for full neural training and visual-model planning if the runtime probe still passes.

## Open risks

- A bounded exact-dynamics CEM may not meet the development gate. If so, report the planner limitation and stop before learned-model evaluation.
- Even a passing development and final oracle gate does not make a 20-start, five-seed MountainCar result general evidence about JEPA methods.
- The simple synthetic renderer makes state decoding easy and cannot support claims about natural video or robot cameras.
- A state probe gives the latent planner access to a task-state readout. It is a transparent reward adapter for this synthetic task, not a generally available robot reward function.
