# Tutorial Evaluation Protocol

**Reviewed**: 2026-10-16

Use this protocol for new measurements and when a lesson makes a performance comparison. It does not certify results that were produced before this protocol. Paper results and local results must stay separate.

## 1. Choose a report level

| Level | Minimum record | Claims allowed |
| --- | --- | --- |
| Smoke | One short run; record whether setup and execution worked. | The code path ran in this environment. No performance claim. |
| Exploratory | Record all runs and settings. State which choices changed after looking at results. | Describe these runs only. Do not claim a general improvement. |
| Comparison | Fix the protocol before the final runs. Use at least five independent training seeds when training is stochastic, and at least 20 fixed evaluation episodes per trained policy where episodes apply. Report every run. | Compare the named methods on the named task and conditions. Treat intervals from a small number of training runs as uncertain. |

The seed and episode counts are project defaults for teaching comparisons, not universal field standards. More runs may be needed. If compute limits prevent them, keep the result exploratory and state the limit. For a frozen checkpoint, record its exact source and hash or revision; training seeds are not applicable, but the evaluation data or episode seeds still are.

## 2. Define the task before running

Record:

- Task, environment or dataset name, version, wrappers, observation and action space, and episode end rules.
- Primary outcome and its exact calculation. For control, include episode return and task success when the environment defines success. State the maximum episode length.
- Training, validation, and evaluation data or reset seeds. Keep evaluation cases out of training and model selection.
- Model, objective, checkpoint, baseline, and the changes between them.
- Planned training seeds, evaluation episodes, and any exclusions. Do not remove failed runs after seeing their scores; explain setup failures separately from low scores.

For a planning lesson, also record rollout horizon, action candidates or optimizer steps, replanning frequency, feedback mode, goal distance, and compute per decision. State whether the plan uses predicted or observed states.

## 3. Use fair comparisons

- Name a baseline that answers the lesson's question. Examples include a random or scripted policy, a standard controller, a raw-observation model, or the existing method with one component removed.
- Use the same task definition and evaluation episodes for methods in a comparison. If runs can be paired, preserve the pairing in the analysis.
- Give each method the same data split and a stated compute or training budget. Report differences in budget instead of hiding them.
- Select checkpoints and hyperparameters using training or validation data only. Do not select the best evaluation run or tune against the final evaluation set.
- For representation learning, state the downstream split, probe type, fitting procedure, and metric. A visualization such as t-SNE is not a performance metric.

## 4. Report variation

Show individual training-run results. For one task, report the mean and median across runs and a 95% interval across independent training runs. For each run, state how its evaluation episodes were combined into a score. When methods share evaluation cases, report paired per-case differences when they match the question, and preserve that pairing when estimating the interval. Do not pair unrelated training runs only because they use the same seed number. State the interval method and number of runs; do not treat episodes from one trained policy as independent training runs.

For a suite of tasks, report task-level scores and an aggregate that preserves variation across tasks and runs. Interquartile mean and stratified bootstrap intervals are established options in the RLiable evaluation work. Do not replace per-task results with one aggregate. For a small tutorial with too few runs for stable intervals, show the raw values and label the interval as uncertain rather than implying statistical certainty. [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://arxiv.org/abs/2108.13264)

## 5. Record data, model, and compute provenance

Include the data and checkpoint source, revision or identifier, access date, license or access limits, and any filtering or preprocessing. Record software versions, operating system, CPU/GPU, precision, training time, evaluation time, and peak memory when available. Keep downloaded data, weights, credentials, and private paths out of version control.

If a source, license, checkpoint, or dataset is unavailable, mark it unavailable. Do not substitute an unverified model or data source and retain the original claim.

## 6. Keep claim status explicit

Use one of these labels in lesson text and result tables:

- **Paper-reported**: stated by a cited source and limited to that source's setup.
- **Locally measured**: produced by this repository; include the task, settings, and run level.
- **Illustrative**: a toy or synthetic example that teaches a concept but does not measure the cited method.
- **Not run**: no local execution evidence is available.

Do not describe a paper-reported result, a smoke run, or an illustrative example as a reproduced result.

## 7. Match the claim to the evidence

State the evidence needed for each claim. Feature quality, visual plausibility, or open-loop prediction error does not by itself show that a model responds correctly to changed actions or helps a policy choose actions. A planning-success result does not show an advantage over a direct policy unless both methods use matched tasks, evaluation cases, and stated budgets.

For decision-making claims, consider separate measurements for:

- **Prediction**: held-out state or feature prediction under recorded actions.
- **Intervention**: whether predictions change in the right way when the action changes from the same starting state.
- **Closed-loop use**: success or return when the planner executes actions and replans from observed states.
- **Decision benefit**: policy-ranking agreement or improvement over a matched direct-policy or simple-control baseline.
- **Reliability**: calibration, failure cases, and whether the planner finds actions that exploit model errors.

This list is a project reporting aid, not a universal benchmark or a single ordered score. The decision-making evaluation literature discusses related evidence levels and metrics; select only those needed for the paper claim. [Yu et al., *How Should World Models Be Evaluated for Embodied Decision-Making?*](https://arxiv.org/abs/2606.15032); [Jena et al., *Do World Models Make Better Robots?*](https://arxiv.org/abs/2609.29669).

## 8. Paper-reproduction protocol

Use this section when the goal is to repeat a published finding. Select one result before running and name its exact table, figure, or claim. Record the expected metric and a tolerance or decision rule before viewing local results.

- Record the paper version, official code revision, checkpoint and data sources, access date, license limits, model and data scale, preprocessing, training recipe, evaluation split, metric, seeds, and compute.
- Match the paper's task and key settings. If data, model scale, training budget, hardware, or evaluation changes, list the changes before the run and label the work a **scaled reproduction attempt** or **partial reproduction attempt**.
- Use the paper's evaluation metric and report all runs, including failures. Do not select hyperparameters or checkpoints with the final evaluation set.
- Compare the reproduced method with the paper's stated baseline on the same data split and evaluation cases. Keep paper-reported values and local measurements in separate columns.
- A successful local reproduction supports only the selected result under the recorded setup. It does not validate every claim in the paper. A mismatch is evidence about this reproduction attempt, not a general refutation.
- Use the paper's run count when stated. If it is not stated, use the project default of at least five independent training seeds for stochastic training. If resource limits prevent this, report the work as exploratory or partial and give the reason.

Do not call a synthetic clip, toy environment, loss smoke test, or reduced-scale run a reproduction of a benchmark finding. Use **illustrative** or **scaled reproduction attempt** as appropriate.

## 9. Result record

Use a compact results table and include these fields in the table or its adjacent run record:

| Method | Task and version | Data or checkpoint | Training seeds | Evaluation episodes | Primary metric | Variation | Compute | Claim status |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |

The report may keep shared task, data, runtime, and claim details in a nearby run-conditions section instead of repeating them in every row. Keep each field explicit and easy to find. Link or attach raw per-run values when they are available. Include missing runs and explain why they are missing.

For the retained episode records from Notebooks 20 and 21, run `uv run --locked --no-sync python scripts/validate_episode_evidence.py` from the repository root. The check verifies the two CSV schemas, row counts, group and pairing coverage, matching start states across methods, statuses, and field values. A pass confirms record integrity only. It does not rerun either notebook or validate a scientific claim.

## 10. Current-project status

The protocol has these current applications:

- **Notebook 09** compares patch correspondence from the official V-JEPA 2.1 checkpoint with a mean-RGB baseline on one deterministic synthetic clip. The fresh CPU-kernel repeat verifies the checkpoint hash, finite feature shape, retrieval output, runtime, and memory. Both methods retrieve a patch one grid cell from the known location; there is no pass threshold. This is an illustrative single-clip measurement, not a general feature-quality result or paper benchmark. See the [run record](evidence/vjepa21-local-run.md).
- **Notebook 14** is an illustrative MountainCar analogue. It reports 60 held-out action interventions, 20 paired starts, paired outcome intervals, actual model calls, and elapsed time. Both methods had the same time cap, but actual compute differed; it makes no matched-compute advantage claim. The deterministic local model and one evaluation set do not establish training-run variation or a JEPA result.
- **Notebook 16** is an exploratory simulator analogue. It evaluates one fitted fixed-feature model on 20 paired starts and 60 held-out action interventions. Its episode intervals do not stand in for variation across trained models. It is not a JEPA implementation or a paper reproduction.
- **Notebook 17** is a five-run state-based analogue. It uses 20 paired starts per trained model and reports 95% Student-t intervals across run-level means. Its random-action and planner-shift coverage checks and short selected-plan gaps are not JEPA or paper results.
- **Feature 013** records one full 35-epoch run each for LeHami and LeWM, using the official configurations, verified data split, and K=0 probe. The results are a partial reproduction attempt; seeds 1 and 2 remain open, so no three-seed estimate is available.
- **Notebook 15** passed a full CPU-kernel run of its oscillator example. This confirms local execution only; it is illustrative and does not measure the LeHami model.
- **Notebook 20** compares raw-action and policy-noise sampling in a hand-authored toy task. It uses 100 paired starts across five evaluation-seed groups, equal candidate counts and horizons, a direct state-feedback reference, actual model-call counts, selected-action prediction error, and policy-support checks. The model error is constructed, and seed variation is not training variation. It is illustrative and does not test LeWAM; its 300 rows are retained in the [episode CSV](evidence/policy-noise-episodes.csv).
- **Notebook 21** compares simplified RRT, KPIECE, and CEM on a hand-authored point-mass detour task. It uses 100 shared starts across five groups, the same maximum of 256 one-step transition evaluations per planning decision, closed-loop execution, and reports success, steps, calls, runtime, group variation, and missing/error counts. It completed 300 episodes with no missing or errored records. Its known transition function is illustrative; it does not test DINO-WM, JEPA, or the paper's result. All 300 rows are retained in the [episode CSV](evidence/sampling-based-episodes.csv).
- **Notebook 22** is a five-seed synthetic visual-JEPA study with 20 held-out intervention starts and 20 shared starts per trained model seed. It uses a VISReg-style temporal-prediction adaptation, not the paper's multi-view image objective. Its 0/100 uniform-start and 100/100 momentum-prior planner outcomes do not isolate JEPA value because the same handcrafted controller succeeds on 20/20 starts. Keep its results as historical evidence; do not combine them with Notebook 23.
- **Notebook 23** adds a separately preregistered comparison with a development-tuned and held-out-gated oracle planner. Its 660 episode rows passed validation. The model and data seeds are paired, the direct controller is handcrafted, a prior-only CEM arm is absent, and all five JEPA fits triggered low spread. The JEPA outcomes remain exploratory and the claim status partial. The built-in bootstrap interval is supplemented by a corrected shared-start analysis because the original implementation does not preserve the pairing during start resampling. See the [v1 report](reproductions/visual-jepa-decision-study-v1.md).
- **Visual JEPA action-fidelity v2** is a separate validation-only study, not a paper reproduction. Its frozen protocol completed 15 fits across three JEPA objectives and positive controls. The exact simulator and handcrafted visual state reference confirmed the observation and label paths. The supervised visual control and all nine JEPA fits failed their gates; no objective was selected. The independent test split, rollouts, and planning were not opened. Report this as a local negative diagnostic under the tested recipe, not as a general JEPA result. See the [protocol](reproductions/protocols/visual-jepa-action-fidelity-v2.md), [technical report](TECHNICAL_REPORT.md), [root-cause report](reproductions/visual-jepa-action-fidelity-v2-investigation.md), and [retained evidence](evidence/visual-jepa-action-fidelity-v2-evidence-manifest.json).

These examples do not retroactively certify earlier results. No exact paper reproduction or matched JEPA-versus-direct-policy comparison is complete. The report-field audit is in [evaluation-protocol-audit.md](evidence/evaluation-protocol-audit.md). See the [coverage map](CURRICULUM_COVERAGE.md) for current evidence limits and the [roadmap](ROADMAP.md) for future lessons.
