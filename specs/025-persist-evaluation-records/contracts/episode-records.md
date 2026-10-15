# Episode Record Contract

## Files

- `docs/evidence/policy-noise-episodes.csv` is produced by Notebook 20.
- `docs/evidence/sampling-based-episodes.csv` is produced by Notebook 21.

## Shared rules

- UTF-8 CSV with LF line endings, one header row, and one row for each planned method/start pair.
- Preserve the field order in [data-model.md](../data-model.md).
- Do not drop failed, unsuccessful, or captured-error rows.
- Use an empty CSV field for an inapplicable `None` value. Do not substitute zero.
- Check the row count before writing and re-read the file to verify the number of rows after writing.
- Do not include paper-reported values, downloaded artifacts, or private data.

## Notebook 20

- Exactly 300 data rows.
- Unique `(evaluation_seed, start_id, method)` keys.
- Each of 100 paired starts has one row for each of the three methods.

## Notebook 21

- Exactly 300 data rows.
- Unique `(group_seed, paired_start, method)` keys.
- Each of 100 paired starts has one row for each of the three planners.
- Keep measured runtime as a raw value; it is not expected to match across executions.
