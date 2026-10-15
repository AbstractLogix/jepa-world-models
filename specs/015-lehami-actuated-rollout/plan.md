# Implementation Plan: LeHami Seed-0 Actuated Rollout Evaluation

**Branch**: `015-lehami-actuated-rollout` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/015-lehami-actuated-rollout/spec.md`

## Summary

Evaluate the existing seed-0 LeHami and LeWM checkpoints with the authors' pinned standard evaluator, the official generated actuated-pendulum data and split, the recorded actions, horizon 76, and the 25-neighbor K=1 probe frozen across rollout steps. Store raw output outside the project, summarize the held-out metrics and run conditions, then update Notebook 15 and project alignment. Do not train additional seeds.

## Technical Context

**Language/Version**: Python 3.13.14 in the existing temporary evaluation runtime  
**Primary Dependencies**: Existing official LeHami checkout and its derived locked environment; no project dependency changes  
**Storage**: Existing HDF5 dataset and checkpoints under `/tmp`; unique output directories under the official checkout's results folder  
**Testing**: Run the official evaluator for both models and check raw per-step outputs; no new test suite  
**Target Platform**: Current Linux CUDA host  
**Project Type**: Research evaluation, tutorial notebook, and evidence documentation  
**Performance Goals**: Complete two evaluation-only runs without training  
**Constraints**: Use the pinned source revision, official seed-42 episode manifest, verified data SHA-256, test-split metrics, recorded action sequence, K=1 25-neighbor probe, and horizon 76. Use unique output names. Do not alter or delete existing temporary artifacts. Keep local one-seed evidence separate from the paper's three-seed statistics.  
**Scale/Scope**: Two existing checkpoints, seed 0; no new training, planner evaluation, Push-T transfer, or additional seeds.

## Constitution Check

- Teach from evidence: Pass. Use the paper, official code, and saved local outputs as separate sources.
- Label examples honestly: Pass. This is a one-seed local rollout evaluation, not a full paper reproduction.
- Make the learning path usable: Pass. Notebook 15 displays saved measurements and does not need the research environment to run.
- Measure before claims: Pass. Report exact task, probe, horizon, split, seeds, and compute.
- Keep changes small: Pass. Reuse existing checkpoints and official evaluator; no new packages or model code.

## Research Decisions

See [research.md](research.md). The official standard evaluator fits its K=1 probe on training data and applies it frozen at each rollout step. The paper's Figure 15 reports three-seed statistics. The local report will retain each one-seed curve separately.

## Design

1. Verify the official source revision, checkpoint hashes, dataset hash, manifest, runtime, and output-name availability.
2. Run the official standard evaluator once for `a0q` and once for `b0s`, using CUDA and horizon 76. Use distinct `--exp-name` values and do not pass `--zero-action` or `--skip-extra-probes`.
3. Read the resulting raw `probe_metrics.csv` and `summary.json`. Report held-out `mae_omega` from the K=1-fitted 25-NN columns at K=1, 5, 10, 20, 30, 50, and 76 when present. Preserve all per-step files outside the repository.
4. Record the full commands, hashes, machine, environment, run times, sample counts, results, and limits in the existing reproduction report.
5. Add a compact comparison to Notebook 15 and update the roadmap, coverage map, and project task records. State that this is a prediction probe, not planning evidence.

## Project Structure

```text
docs/reproductions/lehami-actuated-velocity.md
docs/CURRICULUM_COVERAGE.md
docs/ROADMAP.md
notebooks/15_lehami_symplectic_dynamics.ipynb
specs/001-project-vision/tasks.md
specs/001-project-vision/alignment.md
specs/001-project-vision/remaining-work.md
specs/015-lehami-actuated-rollout/
```

The raw evaluator output and checkpoints remain outside the repository. No public API or new shared source module is needed.

## Validation Strategy

- Confirm the source revision and both checkpoint hashes before evaluation.
- Confirm the dataset SHA-256 matches the shipped manifest before evaluation.
- Run the authors' standard evaluator on CUDA for each model and confirm exit status zero.
- Check both raw output schemas, test split row counts, finite values, selected horizons, and KNN=25 run metadata.
- Confirm the reported local results use seed 0 only and remain separate from paper mean ± standard deviation.
- Analyze and converge the feature against the run outputs and updated project records.

## Complexity Tracking

No constitution violations or added project complexity.
