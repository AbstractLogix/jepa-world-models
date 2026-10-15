# Data Model: LeWAM RoboTwin Preflight

This feature records readiness evidence. It does not define an evaluation result schema.

## Candidate

| Field | Meaning |
| --- | --- |
| paper_claim | Paper-reported finding and source revision. |
| evaluation_scope | Proposed released-checkpoint task, setting, and case list. |
| run_status | `not_run`, `partial`, or `measured`. |

## Source record

| Field | Meaning |
| --- | --- |
| source | Direct paper, repository, model, or simulator URL. |
| revision | Paper version, code commit, model revision, or review date. |
| finding | The source fact used to define scope or a gate. |

## Readiness gate

| Field | Meaning |
| --- | --- |
| gate | Terms, platform, runtime, storage, or fixed-case requirement. |
| status | `pass`, `fail`, or `unverified`. |
| evidence | Observed value or source citation. |
| action | What must happen before work can proceed. |

## Run evidence

No run evidence exists for this feature. A future report must add the exact task, setting, case-list hash, checkpoint revision/hash, runtime, raw outcomes, deviations, and claim limit before it can change `run_status`.
