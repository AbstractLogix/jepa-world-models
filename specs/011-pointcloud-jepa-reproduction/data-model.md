# Data Model: Point-Cloud JEPA Push-T Reproduction

## Evaluation asset

An asset record identifies one pinned item:

- Asset kind: code, dataset, or checkpoint.
- Repository and revision.
- File path, byte size, SHA-256, and license.
- Local verification state: verified, missing, or mismatch.

The dataset record is the Push-T v1 archive. The checkpoint records are Point-LeWM and Point-Delta-JEPA from the same model revision.

## Evaluation seed

Each paper seed selects 50 start-goal windows. The fixed list is `1, 25, 35, 36, 40, 42, 65, 80, 83, 99`. Both methods use the same seed and windows. Each seed produces one success rate per method.

## Episode outcome

Each evaluation attempt records:

- Method and paper seed.
- Episode index from 1 to 50.
- Start-goal window identity when emitted by the official evaluator.
- Success flag.
- Exit state or error if an episode does not complete.

## Reproduction summary

For each method, the summary contains ten per-seed success rates, their mean and standard deviation across seeds, the paper's reported mean and standard deviation, the one-standard-deviation comparison label, and any deviations. Episode-level variation is not substituted for seed-level variation.

## Artifact storage

The upstream checkout, 4.62 GB v1 archive, selected checkpoints, raw logs, and optional media stay outside the repository. The project retains only `docs/reproductions/point-cloud-pusht.md`.
