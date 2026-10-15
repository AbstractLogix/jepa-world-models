# Data Model: LeHami Symplectic Dynamics Tutorial

This feature has no persistent application data. The entities below describe evidence shown in the notebook.

## Entities

### Hamiltonian Map Example

- `hamiltonian`: the separable energy function used by the teaching example.
- `step_size`: positive integration step used to construct a leapfrog map.
- `map_matrix`: the linear transformation over `(q, p)` for the selected step.
- `determinant`: the map's signed area and volume scale.
- `polygon_area_before` and `polygon_area_after`: shoelace areas used to check an example region.
- `numeric_tolerance`: maximum accepted absolute error from determinant one and area ratio one.

Validation: use float64; use a closed polygon with at least three distinct vertices; report absolute errors; do not use the toy map as a LeHami checkpoint or a paper result.

### Probe Evidence

- `model`: LeHami `a0q` or LeWM `b0s`.
- `source`: paper Table 1 or project reproduction report.
- `probe`: 25-neighbor K=0 anchor probe on the actuated-pendulum task.
- `seed_count`: three in the paper row; one in the local row.
- `mae_omega_rad_per_second`: angular-velocity MAE in radians per second.
- `status`: paper-reported or locally measured partial evidence.

Validation: keep paper and local rows separate. A local row must show seed 0 and must not be labeled a mean or standard deviation.

### Evidence Scope

- `guarantee`: a structural property of the leapfrog map.
- `measured_claim`: a specific probe measurement and its run conditions.
- `unmeasured_claims`: rollout quality, planning value, and physical-system validity in this feature.

Validation: a structural guarantee does not fill in an unmeasured empirical claim.
