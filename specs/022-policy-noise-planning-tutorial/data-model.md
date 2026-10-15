# Data Model: Policy-Noise Planning Tutorial

## Toy task

- `state`: scalar position `x` in `[0, 1]`.
- `goal`: target position `1.0`.
- `start`: a seeded position sampled from `[0, 0.2]`.
- `episode_limit`: at most eight executed decisions; success is `x >= 0.98`.

## Toy policy and dynamics

- `policy_noise`: scalar `epsilon` sampled from a standard normal distribution.
- `policy_action`: `drive = 0.5 * sigmoid(epsilon)` and `auxiliary = 0.5 * drive`. This hand-authored map defines the toy policy's declared action support.
- `raw_action`: independently sampled `drive` and `auxiliary` values in `[0, 0.5]`; some pairs fall outside the policy's declared support.
- `environment_step`: clip `x + drive` to `[0, 1]`; the auxiliary component has no effect.
- `forward_model_step`: clip `x + drive + 4 * max(auxiliary - 0.5 * drive, 0)` to `[0, 1]`. The final term is an intentional, hand-authored model error outside the policy support.

## Candidate plan and report record

- `candidate_plan`: three action chunks sampled in either raw-action space or policy-noise space.
- `candidate_budget`: 128 candidate plans per decision for each planner.
- `plan_score`: squared predicted terminal goal distance plus a 0.02 action-energy penalty.
- `episode_record`: method, evaluation seed, paired start id, start position, success, executed steps, model-call count, final position, and first-action prediction error.
- `seed_summary`: success rate and mean steps for one of five groups of 20 paired starts. These groups measure evaluation-case and planner-sampling variation, not training variation.

The notebook's scores and actions are synthetic teaching data. They do not represent LeWAM policy outputs or paper measurements.
