# Local Evidence: Policy-Noise Planning

**Status**: Illustrative toy experiment. This is not a LeWAM implementation or paper reproduction.
**Executed**: 2026-10-15 in a fresh CPU Jupyter kernel.

## Source and paper claim

The v1 [LeWAM diffusion-steering paper](https://arxiv.org/abs/2610.12407) moves a sampling-based planner's decision variable from action chunks to Gaussian noise consumed by a flow-matching policy. It decodes actions at imagined states, scores the final predicted latent against a goal, and executes the first chunk before replanning. The paper also projects noise proposals onto the policy sampler's typical set. Its closed-loop evaluation reports results across five training seeds and 50 evaluation seeds on six tasks. The authors report that action-space MPC performs poorly in their setup and policy-noise planning improves closed-loop performance. Those are paper-reported findings.

The local lesson does not use the paper's code, model, data, flow-matching policy, typical-set projection, CEM/MPPI/CMA-ES updates, or robot tasks. The v1 arXiv record reviewed for this lesson does not link an official implementation or checkpoint. The paper-scale reproduction remains not run in the [reproduction matrix](../REPRODUCTION_MATRIX.md).

## Local setup and evaluation

- **Task**: Move a scalar position from a start in `[0, 0.2]` to `1.0`; success is position `>= 0.98`, with at most eight executed decisions.
- **Toy policy**: Map standard-normal noise to `(drive, auxiliary)` with `drive = 0.5 × sigmoid(noise)` and `auxiliary = 0.5 × drive`.
- **Toy environment**: Advance by `drive`; ignore `auxiliary`.
- **Toy forward model**: Add `4 × max(auxiliary − 0.5 × drive, 0)` to predicted progress. This is a deliberate, hand-authored error outside the policy's declared action support.
- **Methods**: Best-of-N raw-action sampling, best-of-N policy-noise sampling, and a direct state-feedback rule.
- **Paired evaluation**: Five declared evaluation seeds (`101`, `202`, `303`, `404`, `505`), 20 common start positions per seed, 100 paired starts total. The methods use the same 128 candidate plans per decision, horizon 3, score, and episode limit. No model is trained; seed variation is from sampled starts and planner proposals, not training runs.
- **Metrics**: Closed-loop success, steps, forward-model calls, selected-action prediction error, and fraction of selected planner actions outside the toy policy support. Candidate evaluation counts use the same budget per decision; total calls differ because episodes stop at success or the step limit.
- **Compute**: Python 3.13.14, Linux WSL2, CPU, standard library only. The fresh run used for the saved records completed its evaluation loop in 0.295 seconds. No data or model files were downloaded; the notebook writes a small local evidence CSV.

## Results

Success rate by evaluation seed:

| Evaluation seed | Raw-action planner | Policy-noise planner | Direct control |
| ---: | ---: | ---: | ---: |
| 101 | 40% | 100% | 100% |
| 202 | 60% | 100% | 100% |
| 303 | 50% | 100% | 100% |
| 404 | 55% | 100% | 100% |
| 505 | 40% | 100% | 100% |

Aggregate results; success-rate variation is sample SD across the five evaluation-seed groups:

| Method | Success rate | Mean steps | Mean model calls per episode | Selected-action prediction error | Selected actions outside policy support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw-action planner | 49.0% ± 8.9% | 7.56 | 2,903.0 | 0.215 | 73.6% |
| Policy-noise planner | 100.0% ± 0.0% | 5.83 | 2,238.7 | 0.000 | 0.0% |
| Direct control | 100.0% ± 0.0% | 2.00 | 0.0 | Not applicable | Not applicable |

Across the 100 paired starts, policy-noise planning succeeded where raw-action planning failed on 51 starts; raw-action planning had no success that policy-noise planning missed. Direct control succeeded on all starts. It sees the toy state directly, so this is not a matched learned-policy baseline.

## Interpretation and limits

In this constructed case, the raw-action planner often selected actions outside the toy policy's declared support. The toy model predicted progress from their auxiliary component, but the toy environment ignored it. Policy-noise sampling stayed on the hand-authored policy relation and avoided that specific error. The direct controller was faster and used no model calls.

This result shows the behavior of the chosen toy functions only. The model error was inserted by design; the policy was not trained; the task is one-dimensional; the noise input is a scalar sigmoid map; and the planner uses one random-shooting pass rather than the paper's algorithms. The five evaluation-seed groups do not measure training-run variation. These values do not reproduce or validate the paper's reported results.

The notebook regenerates the run from its fixed seeds and writes all 300 per-episode rows to [policy-noise-episodes.csv](../evidence/policy-noise-episodes.csv). The file records the method, evaluation seed, paired start, success, steps, model calls, final position, selected-action error, and off-support fraction. Blank metric fields mean that a metric does not apply to direct control.
