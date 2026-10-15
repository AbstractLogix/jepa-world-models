# Evaluation Protocol Application Audit

**Reviewed**: 2026-10-15
**Scope**: Report fields and claim labels for Notebook 09, Notebook 14, Notebook 16, Notebook 17, Notebook 20, Notebook 21, and the LeHami comparison; verify the saved or regenerated records available for each.

## Result

The shared protocol fits these seven examples. Each report states its task, method and data, run scope, metric, compute conditions, evidence level, and claim limits. The required fields are sometimes split between a results table and nearby run conditions. Section 9 allows that layout while requiring each field to remain explicit and easy to find. Notebook 09 is a single-clip illustrative measurement, so this audit checks its execution record and claim boundary; it does not treat it as a performance comparison or general feature evaluation.

This audit checks the reports and the saved or freshly regenerated local records available for each example. It does not independently verify the paper's results or the external LeHami output files.

## Evidence reviewed

| Example | Protocol fit | Records checked |
| --- | --- | --- |
| [Notebook 09 report](vjepa21-local-run.md) | Correctly labeled local checkpoint execution on a synthetic clip. It records the input, pinned code and checkpoint hash, output shape, simple baseline, runtime, memory, and the absence of a pass threshold. | The fresh CPU-kernel repeat found the expected checkpoint hash and finite `[1, 8, 24, 24, 768]` features, completed without cell errors, and returned the recorded one-cell correspondence for both methods. |
| [Notebook 14 report](../reproductions/decision-usefulness.md) | Correctly labeled illustrative. It fixes disjoint training, intervention, and evaluation seed groups; reports paired cases, intervention errors, closed-loop outcomes, actual compute, and limits. It states peak memory was not measured. | The saved notebook output has 60 intervention rows and 40 paired episode rows. |
| [Notebook 16 report](../reproductions/action-video-planning.md) | Correctly labeled exploratory. It reports one fitted model, episode-level intervals, and limits on training-run variation and paper claims. | The episode CSV has 60 complete rows: 20 for each of three methods. The intervention CSV has 60 rows. |
| [Notebook 17 report](../reproductions/model-uncertainty-planning.md) | Reports five independent model seeds, disjoint data seed groups, paired evaluation starts, run-level Student-t intervals, compute, and its simulator-analogue limit. | The run CSV has five complete runs; calibration has 60,000 rows; plans have 200 complete rows; episodes have 220 complete rows, including the 20 direct-policy episodes. |
| [LeHami report](../reproductions/lehami-actuated-velocity.md) | Correctly labeled partial reproduction. It records one full seed per model, the official source/configs and data hash, the K=0 metric, runtime, and that seeds 1 and 2 remain open. It does not claim the paper's three-seed estimate. | The report records checkpoint hashes and external raw-output paths. This audit did not read those external files. |
| [Notebook 20 report](../reproductions/policy-noise-planning.md) | Correctly labeled illustrative. It reports paired success, steps, model calls, selected-action prediction error, off-support action fraction, run conditions, and limits. It distinguishes five evaluation-seed groups from training-run variation. | A fresh CPU-kernel run completed all nine cells. It regenerated 300 in-memory episodes on 100 shared starts across five groups. Raw per-episode outputs are not saved. |
| [Notebook 21 report](../reproductions/sampling-based-planning.md) | Correctly labeled illustrative. It reports paired task outcomes, steps, transition calls, runtime, group variation, run conditions, and the paper boundary. It distinguishes evaluation-group variation from training-seed variation. | A fresh CPU-kernel run completed all 12 cells and regenerated 300 in-memory episodes on 100 shared starts. Each method completed 100 planned episodes with zero missing or errored records. Raw episode rows are not saved. |

The checked CSV records contain no missing or failed status rows. Notebook 14's saved tables contain all 60 interventions and 40 paired episodes. Notebook 20's regenerated run used all 100 paired starts for each method. The LeHami single-seed result remains partial evidence; no local three-seed mean or standard deviation is available.

## Limits and pending work

- These checks support the completeness and labeling of the reviewed local records. They do not prove that a model or result matches a paper.
- Notebook 16 and Notebook 17 are local simulator analogues, not paper reproductions.
- Notebook 14's paired intervals describe its 20 fixed MountainCar starts. They do not describe variation across model training runs, and its shared time cap did not produce matched actual compute.
- Notebook 20 deliberately inserts a forward-model error outside a hand-authored policy support. Its result demonstrates only that toy setup; it does not validate diffusion steering or any paper claim.
- Notebook 21 uses the same hand-authored transition rule for planning and execution. Its outcome differences show only this toy task and these simplified planners; they do not validate the paper's latent-space search or runtime claims.
- LeHami seeds 1 and 2 remain pending under the current run scope.
- The Push-T archive transfer and evaluation remain scoped and pending. This audit did not start them.
