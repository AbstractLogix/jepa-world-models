# Evaluation Protocol Application Audit

**Reviewed**: 2026-10-15
**Scope**: Report fields, claim labels, and saved-record completeness for Notebook 14, Notebook 16, Notebook 17, and the LeHami comparison.

## Result

The shared protocol fits these four examples. Each report states its task, method and data, run scope, metric, compute conditions, evidence level, and claim limits. The required fields are sometimes split between a results table and nearby run conditions. Section 9 now allows that layout while requiring each field to remain explicit and easy to find.

This audit checks the reports and tracked records. It does not rerun training, evaluation, or notebooks. It does not independently verify the paper's results or the external LeHami output files.

## Evidence reviewed

| Example | Protocol fit | Records checked |
| --- | --- | --- |
| [Notebook 14 report](../reproductions/decision-usefulness.md) | Correctly labeled illustrative. It fixes disjoint training, intervention, and evaluation seed groups; reports paired cases, intervention errors, closed-loop outcomes, actual compute, and limits. It states peak memory was not measured. | The saved notebook output has 60 intervention rows and 40 paired episode rows. |
| [Notebook 16 report](../reproductions/action-video-planning.md) | Correctly labeled exploratory. It reports one fitted model, episode-level intervals, and limits on training-run variation and paper claims. | The episode CSV has 60 complete rows: 20 for each of three methods. The intervention CSV has 60 rows. |
| [Notebook 17 report](../reproductions/model-uncertainty-planning.md) | Reports five independent model seeds, disjoint data seed groups, paired evaluation starts, run-level Student-t intervals, compute, and its simulator-analogue limit. | The run CSV has five complete runs; calibration has 60,000 rows; plans have 200 complete rows; episodes have 220 complete rows, including the 20 direct-policy episodes. |
| [LeHami report](../reproductions/lehami-actuated-velocity.md) | Correctly labeled partial reproduction. It records one full seed per model, the official source/configs and data hash, the K=0 metric, runtime, and that seeds 1 and 2 remain open. It does not claim the paper's three-seed estimate. | The report records checkpoint hashes and external raw-output paths. This audit did not read those external files. |

The checked CSV records contain no missing or failed status rows. Notebook 14's saved tables contain all 60 interventions and 40 paired episodes. The LeHami single-seed result remains partial evidence; no local three-seed mean or standard deviation is available.

## Limits and pending work

- These checks support the completeness and labeling of the reviewed local records. They do not prove that a model or result matches a paper.
- Notebook 16 and Notebook 17 are local simulator analogues, not paper reproductions.
- Notebook 14's paired intervals describe its 20 fixed MountainCar starts. They do not describe variation across model training runs, and its shared time cap did not produce matched actual compute.
- LeHami seeds 1 and 2 remain pending under the current run scope.
- The Push-T archive transfer and evaluation remain scoped and pending. This audit did not start them.
