from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_DIR = Path(
    "/media/sf_OdinShare/MachineLearningCSV/MachineLearningCVE"
)
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

files = sorted(DATA_DIR.glob("*.csv"))
if len(files) != 8:
    raise RuntimeError(f"Expected 8 CSV files, found {len(files)}")

results = []
all_labels = Counter()

for path in files:
    print(f"\nInspecting: {path.name}", flush=True)

    rows = 0
    labels = Counter()
    missing_cells = 0
    infinite_cells = 0
    invalid_numeric_cells = 0
    columns = None

    # Read in chunks to keep memory use manageable.
    for chunk in pd.read_csv(
        path, chunksize=50_000, encoding="utf-8-sig",
        encoding_errors="strict", low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()
        if "Label" not in chunk.columns:
            raise ValueError(f"Label column missing in {path.name}")

        columns = list(chunk.columns)
        rows += len(chunk)

        cleaned_labels = chunk["Label"].astype("string").str.strip()
        labels.update(cleaned_labels.fillna("<MISSING>").tolist())
        missing_cells += int(chunk.isna().sum().sum())

        features = chunk.drop(columns=["Label"])
        numeric = features.apply(pd.to_numeric, errors="coerce")
        invalid_numeric_cells += int(
            (numeric.isna() & features.notna()).sum().sum()
        )
        infinite_cells += int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        )

    record = {
        "file": path.name,
        "rows": rows,
        "columns": columns,
        "labels": dict(labels),
        "missing_cells": missing_cells,
        "infinite_feature_cells": infinite_cells,
        "invalid_numeric_feature_cells": invalid_numeric_cells,
    }
    results.append(record)
    all_labels.update(labels)

    print(f"Rows: {rows:,}")
    print(f"Columns: {len(columns)}")
    print(f"Labels: {dict(labels)}")
    print(f"Missing cells: {missing_cells:,}")
    print(f"Infinite feature cells: {infinite_cells:,}")
    print(f"Invalid numeric feature cells: {invalid_numeric_cells:,}")

report = {
    "data_directory": str(DATA_DIR),
    "total_rows": sum(item["rows"] for item in results),
    "labels": dict(all_labels),
    "files": results,
}
report_path = REPORT_DIR / "dataset_inspection.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nTOTAL LABEL COUNTS")
for label, count in sorted(all_labels.items()):
    print(f"{label}: {count:,}")

print(f"\nReport saved to: {report_path.resolve()}")