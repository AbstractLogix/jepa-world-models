# Quickstart: Calibrate and Stress-Test a World Model

## Requirements

- Python 3.10–3.13 and the project's base dependencies.
- CPU only. No data, model weights, or optional extras are needed.
- Run from the repository root so local CSV paths resolve.

Install the project dependencies if needed:

```bash
poetry install
```

## Execute

Run Notebook 17 in a fresh kernel:

```bash
poetry run jupyter nbconvert --execute --to notebook --inplace notebooks/17_model_uncertainty_planning.ipynb
```

The notebook writes four compact CSV records under `docs/evidence/`: model-run provenance, calibration transitions, selected plans, and closed-loop episodes. The files remain local working-tree evidence and are not staged or committed.

## Check the result

- Confirm training, calibration, random-test, policy-shift, and final-evaluation seeds do not overlap.
- Confirm five independent model runs complete; each learned planner has 20 paired episodes per model run, and the direct policy has one episode per shared evaluation seed.
- Review position and velocity interval coverage and width separately for random-action and policy-shift transitions.
- Review each initial selected plan's predicted return, same-sequence simulator return, and signed gap.
- Confirm all missing or failed rows remain visible. Report mean, median, and the 95% interval across five model runs as uncertain.

The notebook uses a state-based polynomial model in a local simulator. The result is illustrative and does not reproduce a JEPA paper or a physical-robot result.
