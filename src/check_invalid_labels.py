from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path(
    "/media/sf_OdinShare/MachineLearningCSV/MachineLearningCVE"
)

total = Counter()
affected = Counter()

for path in sorted(data_dir.glob("*.csv")):
    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        labels = chunk["Label"].astype("string").str.strip()

        features = chunk.drop(columns=["Label"]).apply(
            pd.to_numeric, errors="coerce"
        )
        invalid_rows = ~np.isfinite(
            features.to_numpy(dtype=float)
        ).all(axis=1)

        total.update(labels.tolist())
        affected.update(labels.loc[invalid_rows].tolist())

print("Label | Total rows | Invalid rows | Remaining rows")
for label in sorted(total):
    print(
        f"{label} | {total[label]:,} | "
        f"{affected[label]:,} | {total[label] - affected[label]:,}"
    )