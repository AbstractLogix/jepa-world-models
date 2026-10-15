# Research: LeHami Symplectic Dynamics Tutorial

**Reviewed**: 2026-10-15

## Findings

- The LeHami paper defines a canonical latent state `(q, p)`, where the position-like state comes from the encoder and the momentum-like state comes from a causal time derivative. It advances that state with a leapfrog step for a separable Hamiltonian `H(q, p, u) = K(p) + V(q, u)`. The paper's guarantee assumes the action stays fixed within each step. The map is symplectic and preserves phase-space volume at every step.
- Symplecticity does not bound rollout error. It preserves phase-space volume, not the shape or physical meaning of a learned representation. The paper also states that energy is not constant under actuation. These limits must appear next to the guarantee, not only in a footnote.
- Table 1 reports the actuated-pendulum K=0 anchor 25-neighbor test probe, averaged over three training seeds: LeHami `0.419 ± 0.045 rad/s`; LeWM `2.508 ± 0.002 rad/s`.
- The official LeHami repository contains the experiment runners, data preparation, split manifests, and a paper-figure notebook. The project reproduction report records one complete seed-0 training and probe for each model: LeHami `0.456258 rad/s`, LeWM `2.504686 rad/s`. Four runs for seeds 1 and 2 remain pending.

## Decisions

### Decision: Teach the structure with a harmonic-oscillator calculation

Use a deterministic, two-dimensional harmonic oscillator to calculate a leapfrog map's Jacobian determinant and the area of a closed polygon before and after the update. Compare it with an explicit-Euler calculation.

**Rationale**: This shows volume preservation with a small CPU-only calculation. It needs no dataset, checkpoint, or training. Explicit Euler is only a numerical contrast; it is not the LeWM baseline or a proxy for the paper's learned model.

**Alternatives considered**: Re-run LeHami training or its official figure notebook. Both add a large external environment and do not improve the learner's ability to isolate the mathematical guarantee. Reuse cached paper figures. That would not let the learner inspect the calculation.

### Decision: Show paper and local probe values as separate evidence

The notebook will report the paper's three-seed statistics and the existing local seed-0 metrics as distinct rows. The numbers are evidence notes, not recomputed outputs of the harmonic-oscillator example.

**Rationale**: The notebook can run without the temporary training outputs, while keeping the source and run count visible. This avoids implying that its local demo reproduced the paper experiment.

### Decision: Keep the tutorial classified as an analogue with partial paper-linked evidence

Do not call the tutorial or paper result reproduced. Do not use the K=0 probe to claim rollout quality, planning value, or physical validity.

**Rationale**: The local experiment has one seed per model and measures only an anchor probe.

## Primary Sources

- Yalovetzky, Balestriero, and LeCun, [LeHami: Symplectic Latent Dynamics for World Models](https://arxiv.org/abs/2610.15267), v1, submitted 2026-10-11. The theory and limitations are in Sections 2 and Appendix C; the actuated-pendulum anchor result is Table 1.
- Authors' [official LeHami repository](https://github.com/rominay/LeHami), released under Apache-2.0. Its README describes the source layout, experiment runners, generated data, and shipped split manifests.
- Current project measurements and conditions: [LeHami actuated-pendulum reproduction report](../../docs/reproductions/lehami-actuated-velocity.md).

The paper source states CC BY 4.0. This tutorial links to the paper and does not copy paper figures, code, or data.
