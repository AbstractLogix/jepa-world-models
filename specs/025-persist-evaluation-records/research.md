# Research: Persist Raw Evaluation Records

**Reviewed**: 2026-10-15

## Finding

The shared [evaluation protocol](../../docs/EVALUATION_PROTOCOL.md) asks reports to link raw per-run values when available. The current [protocol audit](../../docs/evidence/evaluation-protocol-audit.md) says Notebook 20 and Notebook 21 regenerate 300 rows in memory but do not save per-episode records.

## Decision

Write each notebook's existing records as a small CSV in `docs/evidence/`. Keep the fixed seeds, planner settings, aggregation, and illustrative labels unchanged. Include all rows and record the schemas in this feature's data model.

## Source boundary

This is an evidence-retention change for existing toy lessons. It adds no new paper claim or external data. The paper results and the local analogy outcomes remain separate.
