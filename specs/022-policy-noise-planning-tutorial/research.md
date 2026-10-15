# Research: Policy-Noise Planning Tutorial

**Reviewed**: 2026-10-15

## Primary source

- Hegde et al., [LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC](https://arxiv.org/abs/2610.12407), v1, submitted 2026-10-08. The paper describes a decoder-free JEPA world-action model with forward, backward, inverse-dynamics, and policy modes. For planning, it changes the sampling variable from raw action chunks to the policy head's Gaussian noise, decodes each chunk at the imagined state, scores the final predicted latent against a goal latent, and executes the first chunk before replanning. It also projects proposals onto the policy sampler's typical set.
- The paper reports its closed-loop evaluation over five training seeds and 50 evaluation seeds on six tasks. It reports that action-space MPC performs poorly in its setup and that noise-space planning improves performance. These are paper-reported findings, not local results.
- The arXiv page and v1 HTML contain no author-linked code, checkpoint, or benchmark-data repository. The local paper-scale reproduction remains gated in `docs/REPRODUCTION_MATRIX.md`.

## Decision

Create a small, standard-library-only CPU lesson. Use a hand-authored policy map, toy environment, and forward model whose prediction error grows for actions outside the policy's declared action support. Compare raw-action sampling, policy-noise sampling, and a simple direct controller on paired starts with the same planner candidate budget and horizon.

## Rationale and limits

- This isolates the planning-variable idea without copying unavailable paper artifacts or adding dependencies.
- The example deliberately constructs the model error that makes raw-action search fail. That design can teach a mechanism, but it cannot establish that diffusion steering improves real systems.
- The policy map is not trained and is not a diffusion or flow-matching model. The planner will use a bounded random-shooting budget; it will not reproduce the paper's full DS-CEM, DS-MPPI, DS-CMA-ES, latent model, dataset, or benchmark.
- The notebook's variation is across declared evaluation seeds and paired starts. There are no training seeds because no model is trained.
