from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    "/media/sf_OdinShare/MachineLearningCSV/MachineLearningCVE"
)
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

parts = []
removed_invalid = Counter()
expected_columns = None

files = sorted(DATA_DIR.glob("*.csv"))
if len(files) != 8:
    raise ValueError(f"Expected 8 CSV files, found {len(files)}")

for path in files:
    print(f"Reading: {path.name}", flush=True)

    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        labels = chunk.pop("Label").astype("string").str.strip()

        if labels.isna().any() or labels.eq("").any():
            raise ValueError(f"Missing labels in {path.name}")

        features = chunk.apply(pd.to_numeric, errors="coerce")

        original = features["Fwd Header Length"]
        repeated = features["Fwd Header Length.1"]
        equal = original.eq(repeated) | (
            original.isna() & repeated.isna()
        )
        if not equal.all():
            raise ValueError("Repeated header column differs")

        features = features.drop(columns=["Fwd Header Length.1"])
        columns = list(features.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"Column layout differs in {path.name}")

        valid = np.isfinite(
            features.to_numpy(dtype=np.float64)
        ).all(axis=1)
        removed_invalid.update(labels.loc[~valid].tolist())

        clean = features.loc[valid].astype("float64").copy()
        clean["original_label"] = labels.loc[valid]
        clean["target"] = labels.loc[valid].ne("BENIGN").astype("uint8")
        clean["source_file"] = path.name
        clean["source_row"] = clean.index + 1  # Data row, excluding header.
        parts.append(clean)

print("Combining valid records...", flush=True)
data = pd.concat(parts, ignore_index=True)
del parts
feature_columns = expected_columns
valid_rows = len(data)

# Use hashes to locate candidate conflicts, then verify exact values.
print("Checking conflicting labels...", flush=True)
hashes = pd.util.hash_pandas_object(
    data[feature_columns], index=False
)
hash_labels = pd.DataFrame({
    "hash": hashes,
    "target": data["target"],
})
counts = hash_labels.groupby("hash")["target"].nunique()
candidate_hashes = counts[counts > 1].index
candidate_mask = hashes.isin(candidate_hashes)

candidates = data.loc[candidate_mask]
conflicting_mask = pd.Series(False, index=data.index)

if not candidates.empty:
    exact_counts = candidates.groupby(
        feature_columns, dropna=False, sort=False
    )["target"].transform("nunique")
    conflicting_mask.loc[candidates.index] = exact_counts.gt(1)

conflicts = data.loc[conflicting_mask]
conflict_labels = conflicts["original_label"].value_counts().to_dict()
conflict_rows = len(conflicts)

if conflict_rows:
    conflicts.to_pickle(OUTPUT_DIR / "conflicting_rows.pkl")

data = data.loc[~conflicting_mask].reset_index(drop=True)
del candidates, conflicts, hashes, hash_labels, conflicting_mask

# Exact duplicate feature vectors with the same binary target.
print("Removing exact duplicates...", flush=True)
duplicate_mask = data.duplicated(
    subset=feature_columns + ["target"], keep="first"
)
duplicate_labels = (
    data.loc[duplicate_mask, "original_label"].value_counts().to_dict()
)
duplicate_rows = int(duplicate_mask.sum())
data = data.loc[~duplicate_mask].reset_index(drop=True)

# Keep original labels intact, including their existing encoding artifacts.
output_path = OUTPUT_DIR / "cleaned_flows.pkl"
data.to_pickle(output_path)

report = {
    "valid_rows_before_deduplication": valid_rows,
    "invalid_rows_removed_by_label": dict(removed_invalid),
    "exact_conflicting_rows_quarantined": conflict_rows,
    "conflicting_rows_by_label": conflict_labels,
    "exact_duplicate_rows_removed": duplicate_rows,
    "duplicate_rows_removed_by_retained_source_label": duplicate_labels,
    "final_rows": len(data),
    "feature_count": len(feature_columns),
    "final_binary_counts": {
        str(key): int(value)
        for key, value in data["target"].value_counts().items()
    },
    "final_original_label_counts": (
        data["original_label"].value_counts().to_dict()
    ),
    "duplicate_policy": (
        "Keep first record in sorted file order for identical features "
        "and binary target; attack-subtype counts reflect that choice."
    ),
    "features": feature_columns,
}

report_path = REPORT_DIR / "cleaning_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nCLEANING SUMMARY")
print(json.dumps(
    {key: value for key, value in report.items() if key != "features"},
    indent=2,
))
print(f"\nDataset saved: {output_path}")
print(f"Report saved: {report_path}")