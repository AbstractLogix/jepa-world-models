# Data Model: Released LeWAM Push-T Checkpoint Reproduction

## Artifact pin

- Official code repository and commit.
- Model repository and revision.
- Dataset repository and revision.
- Artifact file name, byte size, SHA-256, and declared license.

## Evaluation run

One evaluation mode and seed, containing:

- Mode: reactive, best-of-32, or gradient planner.
- Seed: 42, 0, or 1.
- Goal offset: 25 environment steps.
- Evaluation budget: 50 actions.
- 50 episode outcomes, each with success/failure and any termination details emitted by the official evaluator.
- Software versions, device, runtime, and output paths.

## Summary

For each mode:

- Per-seed success count and rate.
- Mean and standard deviation across the three seed-level rates.
- Paper-reported mean and standard deviation.
- Deviations, missing runs, or errors.

## Reproduction report

A dated Markdown record that links artifact pins, raw output locations, local measurements, paper values, data boundary, compute, and claim status. Generated artifacts remain outside version control.
