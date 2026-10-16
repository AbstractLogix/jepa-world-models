# Data Model: Supervised Visual Positive-Control Study v3

All data are local simulator records. Training and validation data share one data seed but are split by episode identity. Test records use reserved seeds and are generated only if the frozen validation gate opens.

## SeedPair

| Field | Rule |
|---|---|
| `data_seed` | One of 9701, 9702, 9703; generates 20 random-action episodes. |
| `model_seed` | One of 9801, 9802, 9803; paired by index with a data seed. |
| `shuffle_seed` | Stable hash of protocol version, role `action-label-permutation`, and model/data seed pair. |
| `role` | `validation`; test data seeds 9911–9915 are reserved until the representation, one-step, action-fidelity, shuffle-control, completeness, and evidence-lock gates pass. |

## Transition

| Field | Type | Validation |
|---|---|---|
| `frames` | uint8 `[3, 48, 128, 3]` | Index 0 is previous, 1 is current, 2 is next. Only indexes 0 and 1 are model input. |
| `action` | float32 scalar | In [-1, 1]. The action label is shuffled only for the negative-control training copy. |
| `state` | float32 `[position, velocity]` | Current privileged simulator target. Used for training labels and exact diagnostics. |
| `next_state` | float32 `[position, velocity]` | Next privileged simulator target. |
| `episode_id` | int | 0–15 training; 16–19 validation; no episode may cross the split. |

## ValidationContext

One seeded validation row selected without replacement from episodes 16–19. It contains two rendered frames and the corresponding current simulator state. All methods use the same contexts. The triplet expands it to the three fixed actions -1, 0, +1.

## ModelRun

One method and one seed pair. It records status, epoch count, train/validation row counts, device and versions, elapsed time, peak GPU allocation, finite checks, target scales, embedding diagnostics where applicable, raw position and velocity errors, action metrics where applicable, clipping counts, and gate results. Candidate dynamics also require no more than 30 low-spread dimensions, effective rank at least 2.0, and position and velocity linear-probe R² of at least 0.80. Failed and partial runs remain records.

## TrainingCurve

One row per method, seed pair, and epoch. It stores objective loss and raw position/velocity MSE on train and validation partitions. Per-coordinate gradient diagnostics are recorded at the fixed epochs in the protocol.

## ActionTriplet

One row per method, seed pair, validation context, and action. It stores raw predicted position and velocity, simulator next state, action sensitivity, clipping flags, and finite status. Complete contexts have exactly one row for each action -1, 0, +1.

## GateDecision

One row per seed and method, plus a family summary. It records each threshold result and the final held-out access decision. A failed or missing required check means the test stage remains closed.

## EvidenceManifest

The evidence manifest contains a relative repository path, byte size, and SHA-256 for each retained evidence file. The figure manifest contains the same fields for each PNG and SVG. Neither manifest contains a machine-local absolute path, frame tensor, checkpoint, or secret.
