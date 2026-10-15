# Quickstart: Inspect Raw Planning-Lesson Records

Run all cells in Notebook 20 and Notebook 21 from a project checkout. Each notebook writes its episode rows to `docs/evidence/`.

Check the saved row counts with Python:

```bash
uv run python - <<'PY'
import csv
from pathlib import Path

for name in ("policy-noise-episodes.csv", "sampling-based-episodes.csv"):
    path = Path("docs/evidence") / name
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    print(f"{path}: {len(rows)} rows")
    assert len(rows) == 300
PY
```

The reproduction reports describe the columns and link the CSV files. Notebook 20 is a policy-noise toy analogy; Notebook 21 is a state-space planning analogy. Neither file is evidence for the associated paper's results.
