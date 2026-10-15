# Quickstart: PiJEPA CAST Table 2

**Current status: Not run. Preflight blocked.** Do not download CAST or start training/evaluation. Dataset terms, required PiJEPA checkpoints, the exact paper case list, and paper-compatible compute are not verified. The official evaluator also differs from the paper in its default number of policy samples and initial prior standard deviation.

## Gate review

Before a future run, update the [preflight report](../../docs/reproductions/pijepa-cast-planning.md) with evidence for every item:

1. Record authoritative CAST usage terms. If terms remain unknown, stop.
2. Pin every method-specific policy or world-model checkpoint, the V-JEPA-2 encoder, evaluator code, and all revisions and hashes. An encoder alone cannot replace a trained policy or world model.
3. Record the fixed validation-case list, order, hash, and relationship to the paper's evaluation cases. If the paper-case match is not confirmed, any result remains partial.
4. Confirm a compatible compute host and enough storage for the 80.8 GB archives, expanded data, checkpoints, and outputs. Name durable reviewer access for raw evidence outside Git. The current 12 GB RTX 5070 is not qualified against the paper's four-H200 setup.
5. Inspect the pinned official wrapper and record its exact invocation. Select all four methods explicitly: default_mppi (Default MPPI), octo_mppi (PiJEPA), octo_wm (Octo-WM), and octo_mean (Octo). Its reviewed default selects only octo_mean.
6. The paper uses four Octo diffusion samples for the PiJEPA policy prior. The wrapper applies N_OCTO_SAMPLES to every mode, and the paper does not state the sample count for standalone Octo. Confirm the standalone Octo setting before choosing per-method invocations; do not assume one shared value matches every method.
7. Resolve or record the initial-standard-deviation difference. The paper clips the policy-prior standard deviation to [0.01, 0.05]; the current evaluator passes the raw sample standard deviation into the first policy-seeded MPPI iteration and clips only later updates. Keep the result partial if this difference is not resolved.

## Run after all gates pass

Use the official PiJEPA evaluator at the pinned revision after all gates pass. The reviewed wrapper requests 1,000 examples with seed 42 and uses 4 MPPI iterations, 32 samples, 4 elites, and temperature 0.8. Confirm these values against the pinned source before execution. Use separate invocations if the methods need different Octo sample counts, but preserve one fixed case list and order across every invocation. Confirm the standalone Octo sample count and resolve or record the initial-standard-deviation difference before running. Do not rely on the inner evaluator's 50-example default.

Use the paper's training-run count when it states one. Otherwise run at least five independent seeds per stochastic training method. For frozen checkpoints, record the exact source and hash; do not treat evaluation episodes as training seeds. Preserve every run, including failures. Keep raw per-case JSON in durable reviewer-accessible storage outside Git. For trained models, report per-run values, mean, median, and a 95% interval across training runs. For frozen checkpoints, report the per-case distribution and interval, separate from training variation. State the interval method and run count. When methods share cases, report paired per-case differences when they match the question. Record environment versions, hardware, elapsed time, deviations, and missing cases or seeds.

## Completion check

A reviewer must be able to verify that:

- all four methods used the same encoder, task, fixed cases, and stated compute budget;
- the primary metric is ATE XY RMSE in meters and other returned Table 2 metrics are retained;
- paper values and local results are separate;
- all planned seeds and cases have raw records;
- any unconfirmed case-list match, missing seed, or deviation keeps the status `partial` or `exploratory`.

At present there are no local evaluator results. The only activity recorded for this feature is source review.
