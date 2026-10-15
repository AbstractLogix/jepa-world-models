# Qantara CPU inference tutorial evidence

**Status**: Complete for the local teaching analogue; no Qantara paper result is claimed.  
**Date**: 2026-10-15  
**Notebook**: [`18_qantara_three_inference_modes.ipynb`](../../notebooks/18_qantara_three_inference_modes.ipynb)

## Run conditions

- All 12 notebook cells ran in order in a fresh Python 3.13.14 kernel on CPU.
- Platform: Linux 6.18.54.1, WSL2, x86_64.
- The notebook code imports only Python standard-library modules. The execution harness used `nbformat` 5.10.4 and `nbclient` 0.10.4 from the project environment.
- No model, checkpoint, dataset, GPU, external service, or download was used.
- The toy CEM-like search used seed 2026, horizon 3, population 160, 20 elites, and 4 iterations. The BC-like sample used seed 19.

## Observed output

- Noise-edge map printed the forward, policy, video, inverse, and joint training modes.
- The CEM-like toy planner used goal 2.40, selected first action 0.854, predicted terminal state 2.395, and reported toy cost 0.000024.
- The BC-like toy sample was 0.488 and received no goal input.
- The video–inverse toy path predicted next latent 0.550 and recovered action 0.550 without a goal input.
- Fixed-seed repeat checks and goal-blind function-signature checks passed.

These numbers describe only the scalar teaching analogy. They are not Qantara measurements, success rates, or evidence of control performance.

## Limits and next step

The notebook does not train a JEPA, use images or Push-T, load released checkpoints, or execute the paper's evaluation. The optional released-checkpoint run remains gated by the data identity, checkpoint, storage, runtime, and renderer checks in the [preflight report](../reproductions/qantara-preflight.md). See [feature 019](../../specs/019-qantara-multiparadigm/spec.md) for the paper-evaluation scope.
