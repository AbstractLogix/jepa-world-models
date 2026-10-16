# Research Notes: JEPA Action Fidelity v2

## Decision: Keep the task and renderer fixed

**Rationale**: The question concerns the visual representation and action dynamics in Notebook 23. A new simulator or dataset would introduce another cause of failure. V1 uses the synthetic MountainCar renderer, two temporal frames, and continuous actions in [-1, 1].

**Alternatives considered**: Move directly to Push-T or a robot benchmark. Rejected for this milestone because that would add a different dataset, action space, and simulator before resolving the current failure.

## Decision: Diagnose visual information with positive controls

**Rationale**: Exact transitions, privileged state dynamics, a handcrafted visual estimator, and a supervised visual dynamics model separate simulator, observation, and self-supervised objective limits. The supervised visual model uses the same frames, actions, train/validation split, and training budget as the JEPA models.

**Alternatives considered**: Treat the negative JEPA planner result as proof that visual JEPA cannot solve the task. Rejected because v1 did not include a supervised visual dynamics control.

## Decision: Compare only the three existing objective families

**Rationale**: Notebook 23 uses a LeJEPA-style predictive-plus-SIGReg objective. Notebook 22 uses a VISReg-style scale/shape/center temporal-prediction adaptation. A local variance/covariance term provides a small diagnostic control because the repository already has a corresponding helper and one prior validation pilot. Keep the architecture, data, optimizer, epochs, and pairs fixed.

**Alternatives considered**: Add additional papers, architectures, and broad hyperparameter sweeps. Rejected because they would obscure the narrow causal question and increase selection risk.

## Decision: Freeze gates before model training

**Rationale**: V1 reported the low-spread cutoff in implementation but not in its preregistration. V2 sets thresholds for spread, effective rank, probe stability, state error, action effects, clipping, and rollout stability before the first fit.

**Alternatives considered**: Select thresholds after a pilot. Rejected because that would condition the gates on observed outcomes.

## Primary sources

- LeJEPA, arXiv 2511.08544: https://arxiv.org/abs/2511.08544
- VISReg, arXiv 2606.02572: https://arxiv.org/abs/2606.02572
- VICReg, arXiv 2105.04906: https://arxiv.org/abs/2105.04906

These papers motivate objective names only. The local MountainCar experiments do not reproduce their full data, architecture, or training recipes.
