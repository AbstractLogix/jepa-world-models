# Implementation Plan: Matched Visual JEPA Decision Study

**Branch**: `codex/matched-visual-jepa-decisions`<br>
**Date**: 2026-10-15<br>
**Spec**: [spec.md](spec.md)

## Summary

Build a versioned follow-up to Notebook 22. First audit and preregister the methodology. Then tune a bounded exact-dynamics CEM planner on development starts and lock its protocol. Open final learned-model evaluation only after the held-out exact-dynamics gate passes. If either planner gate fails, preserve the results and report the study as gated or partial.

## Design

1. Keep the existing task and renderer from Notebook 22. Do not edit its evidence files or overwrite its notebook outputs.
2. Put deterministic planner, seeding, split, probe, and evidence helpers in `src/jepa_tutorial/decision_study.py`. Keep training orchestration, task protocol, and analysis in `scripts/run_visual_jepa_decision_study.py`.
3. Add `notebooks/23_visual_jepa_decision_study.ipynb` as the learner entry point. It explains the gates and runs the versioned script; model code stays out of notebook cells.
4. Use three model families from the existing primitives: privileged state dynamics, visual reconstruction dynamics, and action-conditioned JEPA. Fit all three on the same generated trajectory group for each of five data/model seed pairs. Use a frozen encoder and linear readout for visual position/velocity. The readout uses training data and validation selection only.
5. Score candidate rollouts in task units. Use predicted position/velocity from the state model or visual probe to compute the same task reward. Retain model-native latent, pixel, and state prediction metrics as separate diagnostics.
6. Compare oracle, state, reconstruction, and JEPA models under uniform and momentum-prior CEM. Include the single direct visual-momentum baseline once per start. State access is marked privileged.
7. Use paired evaluation starts and separate development, probe-test, intervention, evaluation, and smoke seeds. The lock file identifies the script/module hash and selected planner parameters.
8. Write all transient output to `JEPA_RUN_OUTPUT_DIR`. Put only small CSV/JSON evidence and figures in the repository. Do not save weights or image arrays.
9. Generate figures from versioned evidence only. Do not add a package dependency.

## Experiment gates

- `smoke`: CPU-only, reserved smoke seeds, no final cases.
- `dev-search`: exact dynamics only; predeclared tuning starts and search grid; bounded by the development call ceiling.
- `oracle-gate`: locked exact planner only; held-out starts; writes a gate record. A failed result prevents the model stage.
- `full`: requires a valid planner lock and passing oracle-gate record. Trains the five declared seed pairs and collects the matched comparison.

The run script must fail closed on a missing, malformed, stale, or mismatched lock or gate record. Each mode writes to a separate prefix.

## Planner protocol before development tuning

The preregistration at `docs/reproductions/protocols/visual-jepa-decision-study-v1.md` is written before running the search. It sets a staged search: compare horizons, rollout scores, and initialization with a fixed baseline; then compare candidate count and iteration count on the tuning subset; finally verify the selected configuration on disjoint development-confirmation starts. The search is limited by a fixed rollout-transition ceiling. The final selection rule is success count, then environment return, then fewer transition calls, then shorter horizon.

## Study stages

1. Validate exact transition parity with Gymnasium and test CEM invariants on CPU.
2. Run the predeclared development-only search. Save every setting and outcome. Select and hash the planner only when the development confirmation rule passes.
3. Run exact-dynamics planning on the held-out starts. Save those rows first. Continue to model training/evaluation only if at least 18/20 starts succeed.
4. If the gate passes, train all declared model families, fit probes without test labels, then collect interventions and episodes. If a run stops, preserve partial rows and list missing conditions.
5. Validate schemas, seeds, pairing, protocol hashes, planner calls, and status fields. Generate figures, report, and project alignment updates from the retained records.

## Project structure

```text
src/jepa_tutorial/decision_study.py
tests/test_decision_study.py
scripts/run_visual_jepa_decision_study.py
scripts/analyze_visual_jepa_decision_study.py
scripts/generate_visual_jepa_decision_study_figures.py
notebooks/23_visual_jepa_decision_study.ipynb
docs/reproductions/protocols/visual-jepa-decision-study-v1.md
docs/reproductions/visual-jepa-decision-study-v1.md
docs/evidence/visual-jepa-decision-study-v1-*.csv
docs/evidence/visual-jepa-decision-study-v1-*.json
docs/portfolio/figures/visual-jepa-decision-study-v1-overview.png
docs/portfolio/figures/visual-jepa-decision-study-v1-architecture.svg
```

## Validation

- Standard-library unit tests for exact transition parity, deterministic CEM sampling, budget accounting, model-call counts, data splits, probe train/test boundaries, evidence schemas, and fail-closed gates.
- CPU smoke mode with no downloads and no final seeds.
- Ruff, package build, notebook JSON/code-cell compilation, evidence validation, deterministic figure regeneration, Markdown links, and `git diff --check`.
- Full training and final comparisons use the prepared E:-hosted CUDA environment. Record its device, PyTorch/CUDA versions, memory, and actual runtime. Do not present CPU smoke as a CUDA result.

## Constraints and risks

- No changes to Notebook 22 or its evidence.
- No new dependency, external data, checkpoint, or robot simulator.
- The exact planner may fail the gate. This is an acceptable result; it blocks conclusions about model failures.
- The visual state probe makes task-reward scoring possible for latent dynamics, but it is an auxiliary task readout and limits the claim.
- GPU speed may not translate to lower end-to-end time for small models. Report actual timings and calls without a speed claim.
