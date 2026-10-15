# Model Uncertainty and Planning: Local Run

**Run date**: 2026-10-15  
**Status**: Complete local simulator analogue; not a JEPA or paper reproduction

## Scope

Notebook 17 studies two evaluation questions from recent world-model research: whether one-step prediction intervals stay calibrated under planner-induced action shift, and whether a model-selected plan performs as predicted in the simulator. It uses a state-based polynomial Ridge ensemble on MountainCarContinuous. It does not use a JEPA representation, images, public robot data, or pretrained weights.

This is a local analogue of the measures discussed by Yu et al. It does not reproduce their paper's empirical result. The selected-plan gap below is an 8-step open-loop diagnostic. It is not a full policy-level XGap.

## Run conditions

- Python 3.13.14, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, Gymnasium 0.29.1.
- Notebook runtime: nbconvert 7.17.0, nbclient 0.10.4, nbformat 5.10.4, ipykernel 7.2.0.
- Linux WSL2, x86_64, CPU only. No new package, data, or model download was used.
- Five independent model seeds: 50000–50004. Each run used ten random-action training episodes and 2,000 transitions. Each ensemble had five bootstrap members.
- Training episode seed groups were 1000–1009, 1100–1109, 1200–1209, 1300–1309, and 1400–1409. Calibration used 2000–2009; random-action test used 3000–3009; planner-shift test used 3500–3509; final evaluation used 4000–4019. The groups were disjoint.
- The model was a degree-three polynomial Ridge regressor with `alpha=1e-6`. Each component's 90% interval scale was fit on calibration data only.
- CEM used horizon 8, 12 candidates, 3 elites, and 2 updates. Risk-penalized CEM used a fixed coefficient of 1 on ensemble return standard deviation. Both learned methods replanned after every simulator step.
- Each of the five fitted models ran both methods on the same 20 final starts. The direct momentum rule ran once on those same 20 starts.
- The notebook wall time was 104.74 seconds. The five model-run records report 19.69–21.24 seconds each, including model fitting, calibration, and that run's final evaluation.
- Gymnasium 0.29.1 gives MountainCarContinuous a 999-step default limit. This lesson uses a 200-step limit. The result applies to this stated cap.

## Calibration results

Each test interval uses a scale fitted only on the calibration split. Coverage intervals are 95% Student-t intervals across five model-run coverage values. Width is the mean interval width in the simulator's position or velocity units.

| Split | Position coverage (95% interval) | Position width | Velocity coverage (95% interval) | Velocity width |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 0.900 (fit target) | 0.00000459 | 0.900 (fit target) | 0.00000441 |
| Random-action test | 0.907 [0.897, 0.917] | 0.00000376 | 0.903 [0.894, 0.912] | 0.00000362 |
| Planner-shift test | 0.890 [0.822, 0.957] | 0.00000202 | 0.892 [0.819, 0.965] | 0.00000194 |

The calibration CSV has 60,000 component rows: five model runs, three splits, ten episodes per split, up to 200 transitions per episode, and two state components. Test coverage stayed near the nominal 90% level in these runs. The broad planner-shift intervals reflect only five independent model fits; they do not establish coverage on other tasks or models.

## Closed-loop results

Success means that Gymnasium terminated at the goal. Return summaries for learned methods first average the 20 paired starts within each model run. The 95% Student-t intervals then use the five independent model-run means. The direct policy has one fixed set of 20 starts, so its table entry has no interval across training runs.

| Method | Success | Mean return across runs (95% interval) | Median run mean | Mean model calls per episode | Mean episode time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean-return CEM | 0 / 100; 0 / 20 in every run | -1.598 [-1.811, -1.385] | -1.548 | 192,000 | 0.401 s |
| Risk-penalized CEM | 0 / 100; 0 / 20 in every run | -1.599 [-1.809, -1.389] | -1.530 | 192,000 | 0.401 s |
| Direct momentum | 20 / 20 | 91.975 | 92.000 | 0 | 0.0005 s |

Paired return differences used the same 20 evaluation seeds for each model run:

| Learned method minus direct momentum | Mean paired difference (95% interval across five run means) |
| --- | ---: |
| Mean-return CEM | -93.573 [-93.786, -93.360] |
| Risk-penalized CEM | -93.574 [-93.784, -93.364] |

Both learned planners failed to reach the goal under this small candidate and horizon budget. The direct rule reached the goal on all 20 fixed starts. This is a negative result for this local setup, not a general finding about learned world models.

## Selected-plan gaps

The plan CSV contains 200 initial plans. For each plan, the notebook recorded the ensemble-predicted return and the simulator return from the same start state and exact action sequence. The summary uses the five per-run mean gaps:

| Method | Mean gap | Median run gap | 95% interval across runs |
| --- | ---: | ---: | ---: |
| Mean-return CEM | -2.35e-10 | -2.55e-10 | [-5.37e-10, 6.79e-11] |
| Risk-penalized CEM | 4.03e-11 | -3.85e-11 | [-3.44e-10, 4.24e-10] |

These gaps are at numerical precision for this short local rollout. They do not measure a long-horizon optimized-policy gap and do not show that model exploitation is absent in general.

## Evidence files

- [Notebook 17](../../notebooks/17_model_uncertainty_planning.ipynb)
- [Run provenance](../evidence/model-uncertainty-runs.csv): five model seeds, training samples and times, versions, status, and failures.
- [Calibration transitions](../evidence/model-uncertainty-calibration.csv): per-transition predictions, intervals, and coverage.
- [Selected plans](../evidence/model-uncertainty-plans.csv): action sequences, predicted returns, simulator returns, and gaps.
- [Closed-loop episodes](../evidence/model-uncertainty-episodes.csv): all 220 outcomes, including compute and status.
- [Evaluation protocol](../EVALUATION_PROTOCOL.md)

All five model runs, 200 plans, and 220 episodes completed. No row was missing or failed.

## Claim limits

- This model receives the exact two-dimensional simulator state. It is not a visual encoder or JEPA model.
- The task is one deterministic simulator with a local 200-step cap. It does not test other tasks, robot data, physical control, or pretrained world models.
- The plan gap covers only the selected first 8-step open-loop sequence. It is not the paper's full optimized-policy exploitability measure.
- Five model runs give uncertain intervals. The fixed 20-start direct-policy mean is descriptive for these seeds.
- The result does not reproduce a research paper's model, dataset, benchmark, or reported number.

## Research reviewed

- Yu et al., [Decision-Centric Evaluation of World Models](https://arxiv.org/html/2606.15032), discuss policy-induced shift, uncertainty calibration, and model exploitability as separate evaluation dimensions. The paper is a position paper, not a single numeric target for this MountainCar experiment. Its illustrative thresholds are not used here.
- The environment follows the [Gymnasium 0.29.1 MountainCarContinuous implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v0.29.1/gymnasium/envs/classic_control/continuous_mountain_car.py). This lesson changes its episode limit from 999 to 200.
- Song and Cai's [EnvProbe preprint](https://arxiv.org/html/2606.31422) studies budgeted environment probing for language-model agents. It is adjacent research, not evidence about learned embodied dynamics.
