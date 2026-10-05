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

invalid_by_column = Counter()
hash_parts = []
file_reports = []
expected_columns = None

for path in files:
    print(f"\nAuditing: {path.name}", flush=True)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()

    if not df.columns.is_unique:
        raise ValueError("Column names collide after stripping spaces")

    columns = list(df.columns)
    if expected_columns is None:
        expected_columns = columns
    elif columns != expected_columns:
        raise ValueError(f"Column layout differs: {path.name}")

    labels = df.pop("Label").astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError(f"Missing or empty labels: {path.name}")

    suspicious_labels = sorted({
        label for label in labels.unique() if "\ufffd" in label
    })

    numeric = df.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(values)
    invalid_rows = invalid.any(axis=1)

    column_counts = {
        column: int(count)
        for column, count in zip(numeric.columns, invalid.sum(axis=0))
        if count
    }
    invalid_by_column.update(column_counts)

    # Check the repeated header rather than assuming it is redundant.
    header_columns = [
        column for column in numeric.columns
        if column == "Fwd Header Length"
        or column.startswith("Fwd Header Length.")
    ]
    repeated_header_check = {}
    if len(header_columns) > 1:
        first = numeric[header_columns[0]]
        for column in header_columns[1:]:
            other = numeric[column]
            equal = first.eq(other) | (first.isna() & other.isna())
            repeated_header_check[column] = int((~equal).sum())

    # Hash only rows with finite features.
    # Hash matches flag likely duplicates; confirm exact equality
    # before removing records during cleaning.
    valid_features = numeric.loc[~invalid_rows]
    valid_labels = labels.loc[~invalid_rows]
    hashes = pd.util.hash_pandas_object(
        valid_features, index=False
    ).to_numpy()
    binary_labels = valid_labels.ne("BENIGN").astype("uint8").to_numpy()

    hash_parts.append(pd.DataFrame({
        "feature_hash": hashes,
        "binary_label": binary_labels,
    }))

    record = {
        "file": path.name,
        "rows": len(numeric),
        "rows_with_invalid_features": int(invalid_rows.sum()),
        "invalid_cells_by_column": column_counts,
        "repeated_header_mismatches": repeated_header_check,
        "labels_with_replacement_character": suspicious_labels,
    }
    file_reports.append(record)

    print(f"Rows: {len(numeric):,}")
    print(f"Rows with invalid features: {invalid_rows.sum():,}")
    print(f"Invalid cells by column: {column_counts}")
    print(f"Repeated-header mismatches: {repeated_header_check}")
    print(f"Labels containing replacement characters: {suspicious_labels}")

    del df, numeric, values, invalid, valid_features

combined = pd.concat(hash_parts, ignore_index=True)
duplicate_candidates = int(
    combined.duplicated(["feature_hash", "binary_label"]).sum()
)
label_counts = combined.groupby("feature_hash")["binary_label"].nunique()
conflicting_hashes = label_counts[label_counts > 1].index
conflicting_rows = int(
    combined["feature_hash"].isin(conflicting_hashes).sum()
)

report = {
    "files": file_reports,
    "total_rows": sum(item["rows"] for item in file_reports),
    "rows_with_invalid_features": sum(
        item["rows_with_invalid_features"] for item in file_reports
    ),
    "invalid_cells_by_column": dict(invalid_by_column),
    "valid_rows": len(combined),
    "duplicate_candidates_beyond_first": duplicate_candidates,
    "feature_hashes_with_conflicting_binary_labels": len(conflicting_hashes),
    "rows_in_conflicting_hash_groups": conflicting_rows,
}

report_path = REPORT_DIR / "dataset_audit.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nAUDIT SUMMARY")
for key, value in report.items():
    if key != "files":
        print(f"{key}: {value}")
print(f"\nReport saved to: {report_path.resolve()}") 