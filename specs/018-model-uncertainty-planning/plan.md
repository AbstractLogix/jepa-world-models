# Implementation Plan: Calibrate and Stress-Test a World Model

**Branch**: `018-model-uncertainty-planning` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/018-model-uncertainty-planning/spec.md`

## Summary

Add Notebook 17 as a local, state-based analogue for uncertainty calibration and model exploitability. Fit five independently seeded ensembles of polynomial ridge dynamics models on MountainCarContinuous data. Calibrate 90% one-step prediction intervals on separate random-action episodes, check coverage on fresh random and planner-induced transitions, then compare mean-return CEM, a fixed uncertainty-penalized CEM, and a direct momentum policy on paired starts. Record the predicted and simulator return for each selected open-loop plan. The lesson does not use a JEPA representation and does not claim to reproduce a paper.

## Technical Context

**Language/Version**: Python 3.10–3.13, following the project range.
**Primary Dependencies**: Existing NumPy, SciPy, scikit-learn, Matplotlib, and Gymnasium dependencies. Add no packages.
**Storage**: Local simulator data only. Keep raw run records in `docs/evidence/` in the working tree; do not download data or model weights.
**Testing**: Execute the full notebook in a fresh CPU Jupyter kernel. Verify split disjointness, run and episode counts, interval calculations, paired comparisons, and persisted CSV rows.
**Target Platform**: CPU; `MountainCarContinuous-v0`; 200-step local episode cap.
**Project Type**: Educational Jupyter notebook and documentation.
**Performance Goals**: Finish five model runs and the fixed evaluation on CPU with every planned row retained. Record wall time; do not claim a speed advantage.
**Constraints**: Use only the simulator's state and action. No image encoder, pretrained model, robot data, hardware, or extra dependency. Keep training, calibration, random-action test, policy-shift test, and final evaluation seeds disjoint.
**Scale/Scope**: Five independently seeded model runs; five bootstrap members per run; ten episodes per training run; ten episodes each for calibration, in-distribution testing, and policy-shift calibration testing; 20 final evaluation starts. CEM uses horizon 8, 12 candidates, 3 elites, and 2 updates. The risk penalty is fixed to one ensemble return standard deviation.

## Constitution Check

- Evidence: PASS. The research section cites a primary evaluation paper and the official Gymnasium environment documentation.
- Honest labels: PASS. The model is a state-based simulator analogue, not a JEPA model or paper reproduction.
- Usable learning path: PASS. The notebook has a CPU path and uses project dependencies already declared.
- Measured claims: PASS. Splits, metrics, five model runs, 20 paired starts, and compute records are fixed before the final evaluation.
- Small changes: PASS. One notebook, local evidence records, and alignment updates; no abstraction or dependency is added.
- Data and access: PASS. The work uses generated local simulation data only.

**Gate result**: Implementation may proceed. No external data, model, license, or hardware gate is required.

## Experiment Design

### Data and run splits

- Five model runs use ten random-action training episodes each. Run `r` uses reset seeds `1000 + 100r` through `1009 + 100r`, for `r` in 0–4.
- Calibration uses random-action reset seeds 2000–2009.
- In-distribution test uses random-action reset seeds 3000–3009.
- Policy-shift test uses mean-ensemble CEM on reset seeds 3500–3509; collect actual one-step transitions along those policy-generated trajectories.
- Final paired evaluation uses reset seeds 4000–4019 for both learned planners and the direct policy.
- All episodes use a 200-step `TimeLimit`; training and validation data do not include final evaluation transitions.

### Model and uncertainty

For each model run, fit five bootstrap resamples of a degree-three polynomial ridge regressor from `(position, velocity, action)` to `(next_position, next_velocity)`. Ensemble disagreement is a raw spread score, not a probability. For each state component, compute a scale from the 90th percentile of absolute calibration residual divided by ensemble standard deviation. Apply those scales only to the independent random-action and policy-shift test splits. Report interval coverage and mean width for position and velocity separately.

### Planning and exploitability

At each CEM decision, each ensemble member rolls each candidate action sequence forward and accumulates the same predicted goal reward and action cost. Mean-return CEM ranks candidates by the ensemble mean. Risk-penalized CEM ranks them by `mean - standard_deviation` of ensemble returns; this coefficient is fixed before evaluation. Both use the same horizon and candidate budget and execute only the first action before replanning from the simulator state.

For the initial top plan at each final-evaluation start, execute that exact 8-step open-loop sequence in a fresh simulator initialized to the same state. Record its mean-model predicted return, simulator return, and signed selected-plan gap. This is a diagnostic, not the paper's full policy-level XGap; it does not replace the closed-loop evaluation.

### Reporting

Retain per-transition calibration rows, per-plan predicted and actual returns, and per-episode outcomes. Summarize each model run first, then report mean, median, and a 95% Student-t interval across the five independent model runs. Report paired per-start differences for the two learned planners versus the fixed direct-policy outcomes. Label five-run intervals uncertain.

## Project Structure

```text
notebooks/17_model_uncertainty_planning.ipynb
docs/evidence/model-uncertainty-calibration.csv
docs/evidence/model-uncertainty-plans.csv
docs/evidence/model-uncertainty-episodes.csv
docs/evidence/model-uncertainty-runs.csv
docs/reproductions/model-uncertainty-planning.md
docs/ROADMAP.md
docs/CURRICULUM_GUIDE.md
docs/CURRICULUM_COVERAGE.md
docs/REPRODUCTION_MATRIX.md
specs/001-project-vision/{tasks.md,alignment.md,remaining-work.md}
```

**Structure Decision**: Keep the small experiment self-contained in one notebook, matching the existing curriculum. No public runtime API or contract is needed. Raw CSV files remain local working-tree evidence and are not staged or committed.

## Complexity Tracking

No constitution violation, new dependency, or project-wide abstraction is introduced.
