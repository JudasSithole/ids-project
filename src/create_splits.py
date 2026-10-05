from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
DATASET = DATA_DIR / "cleaned_flows.pkl"
SPLITS = DATA_DIR / "split_indices.npz"
SEED = 42

if SPLITS.exists():
    raise FileExistsError(
        "Split file already exists. Keep it unchanged for reproducibility."
    )

data = pd.read_pickle(DATASET)
indices = np.arange(len(data))

train_idx, remaining_idx = train_test_split(
    indices,
    test_size=0.40,
    random_state=SEED,
    stratify=data["original_label"],
)

validation_idx, test_idx = train_test_split(
    remaining_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=data.iloc[remaining_idx]["original_label"],
)

# Verify that the split partitions every row exactly once.
combined = np.concatenate([train_idx, validation_idx, test_idx])
assert len(combined) == len(data)
assert np.array_equal(np.sort(combined), indices)

np.savez_compressed(
    SPLITS,
    train=train_idx,
    validation=validation_idx,
    test=test_idx,
)

# Tie these row indices to this exact dataset file.
digest = hashlib.sha256()
with DATASET.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)

report = {
    "seed": SEED,
    "dataset_sha256": digest.hexdigest(),
    "method": "Random split stratified by original attack label",
    "splits": {},
}

for name, selected in [
    ("train", train_idx),
    ("validation", validation_idx),
    ("test", test_idx),
]:
    subset = data.iloc[selected]
    report["splits"][name] = {
        "rows": len(selected),
        "binary_counts": {
            str(label): int(count)
            for label, count in subset["target"].value_counts().items()
        },
        "original_label_counts": (
            subset["original_label"].value_counts().to_dict()
        ),
    }

REPORT_DIR.mkdir(exist_ok=True)
report_path = REPORT_DIR / "split_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nSplit indices saved: {SPLITS}")