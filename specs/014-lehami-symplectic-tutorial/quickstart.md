# Quickstart: LeHami Symplectic Dynamics Tutorial

## Prerequisites

- Python 3.10–3.13. Use the Jupyter environment described in the repository README to open the notebook.
- A CPU. The notebook does not require a GPU.
- No external data, model weights, credentials, robot hardware, or Python packages beyond the standard library for its calculations.

## Run

Open `notebooks/15_lehami_symplectic_dynamics.ipynb` in Jupyter and run all cells from top to bottom.

## Expected Results

- The leapfrog map has determinant one and preserves the polygon area within `1e-12` in float64.
- The explicit-Euler comparison has a determinant above one. It is a teaching contrast, not the LeWM baseline.
- The evidence table shows paper values of `0.419 ± 0.045` rad/s for LeHami and `2.508 ± 0.002` rad/s for LeWM over three seeds. It separately shows local seed-0 values of `0.456258` and `2.504686` rad/s.
- The conclusion labels the local result partial. Seeds 1 and 2 remain open for each model. The K=0 probe is not a rollout, planning, or physical-validity test.

See the [local run record](../../docs/evidence/lehami-symplectic-tutorial.md) and the [measured reproduction record](../../docs/reproductions/lehami-actuated-velocity.md).
