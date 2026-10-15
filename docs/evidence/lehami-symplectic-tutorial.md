# LeHami Symplectic Tutorial Local Check

**Run date**: 2026-10-15  
**Notebook**: `notebooks/15_lehami_symplectic_dynamics.ipynb`  
**Status**: Full notebook execution completed in a fresh CPU Jupyter kernel; all four code cells passed.

## Execution

- Runtime: Python 3.13.14. Notebook code uses the Python standard library.
- All 16 notebook cells were run in order through `nbclient` in a fresh IPython kernel.
- All four code cells completed. All assertions passed, and the output tables match the measurements below.
- The source notebook has stable IDs for all cells. The check confirms kernel execution; it does not assess a graphical Jupyter front end.
- No research dataset, checkpoint, GPU, or training run was used.

## Measurements

| Leapfrog step size | `|det(M) - 1|` | Polygon area ratio | `|area ratio - 1|` | 20-step error against exact oscillator |
| ---: | ---: | ---: | ---: | ---: |
| 0.2 | 1.110e-16 | 1.0000000000000004 | 4.441e-16 | 0.00334234 |
| 0.8 | 0 | 1.0000000000000007 | 6.661e-16 | 0.416915 |
| 2.2 | 2.220e-16 | 1.0000000000000024 | 2.442e-15 | 9.64245e+06 |

All determinant and area errors are below `1e-12`. The large-step example preserves area but has a large trajectory error. This is a numerical teaching example, not a LeHami model result.

The explicit-Euler comparison produced area ratios 1.04 at `h = 0.2` and 1.64 at `h = 0.8`. It is a contrast between integration rules, not the LeWM baseline.

## Evidence limit

The paper/local K=0 probe measurements shown in the notebook are read from the [actuated-pendulum run report](../reproductions/lehami-actuated-velocity.md). They are not outputs of the oscillator calculation. One local seed per model remains partial evidence; seeds 1 and 2 for each model remain open.
