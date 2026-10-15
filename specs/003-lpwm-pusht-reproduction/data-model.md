# Data Model: Bounded LpWM PushT Reproduction

This model defines the records required by the evidence report. Raw data and checkpoints remain outside the repository.

## Experiment definition

- Paper identifier, version, and selected claim.
- Upstream repository revision.
- Model arm (`sparse` LpWM or `dense` LeWM), predictor (`ltv`), latent dimension (768), link, target distribution, regularization weight, muP learning rate, data sampling, epochs, batch size, and training seed.
- Planner configuration, planner seed, evaluation count, and decision rule.

## Dataset record

- OSF project and file identifier.
- Archive name, exact byte count, expected SHA-256, observed SHA-256, and verification status.
- Access date, local path, and license/terms status.
- Extraction status and root directory expected by upstream code.

## Training run

- Model arm and full command.
- Start/end time, exit status, log path, runtime, device, software versions, and available peak memory.
- Checkpoint path and completion status.
- Failure text if the run does not complete.

## Planning run

- Checkpoint identity and planner seed.
- Episode count, success count, success rate, planner settings, runtime, and status.
- Missing episodes or failure details.

## Reproduction report

- Paper-reported value or range and its source location.
- Local per-seed results, mean, standard deviation, and sparse-minus-dense difference.
- Decision-rule outcome and evidence label.
- Dataset access limits, compute, deviations, failures, and limitations.

Paper-reported statements and locally measured values are separate fields. A missing run has no score; it is not a zero-success run.
