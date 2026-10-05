from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

dataset_path = DATA_DIR / "cleaned_flows.pkl"
split_report = json.loads(
    (REPORT_DIR / "split_report.json").read_text()
)
features = json.loads(
    (REPORT_DIR / "cleaning_report.json").read_text()
)["features"]

digest = hashlib.sha256()
with dataset_path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)

if digest.hexdigest() != split_report["dataset_sha256"]:
    raise ValueError("Dataset changed after splitting")

print("Loading data...", flush=True)
data = pd.read_pickle(dataset_path)

with np.load(DATA_DIR / "split_indices.npz") as splits:
    train_idx = splits["train"]
    validation_idx = splits["validation"]

X = data[features].to_numpy(dtype=np.float32)
y = data["target"].to_numpy(dtype=np.uint8)
labels = data["original_label"].to_numpy()
del data
gc.collect()

if not np.isfinite(X).all():
    raise ValueError("Invalid values appeared during float32 conversion")

X_train = X[train_idx]
y_train = y[train_idx]
X_validation = X[validation_idx]
y_validation = y[validation_idx]
validation_labels = labels[validation_idx]
del X, y, labels
gc.collect()

positive_weight = float(
    np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1)
)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cpu",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=positive_weight,
    n_jobs=2,
    random_state=42,
)

print(f"Training on {len(y_train):,} rows...", flush=True)
started = time.perf_counter()
model.fit(X_train, y_train)
training_seconds = time.perf_counter() - started

del X_train, y_train
gc.collect()

print("Evaluating validation data...", flush=True)
started = time.perf_counter()
probabilities = model.predict_proba(X_validation)[:, 1]
prediction_seconds = time.perf_counter() - started
predictions = (probabilities >= 0.5).astype(np.uint8)

tn, fp, fn, tp = confusion_matrix(
    y_validation, predictions, labels=[0, 1]
).ravel()

per_attack = {}
for label in sorted(set(validation_labels)):
    if label == "BENIGN":
        continue
    mask = validation_labels == label
    per_attack[label] = {
        "rows": int(mask.sum()),
        "detected": int(predictions[mask].sum()),
        "recall": float(predictions[mask].mean()),
    }

model_path = MODEL_DIR / "xgboost_baseline.ubj"
model.save_model(model_path)

metadata = {
    "features": features,
    "threshold": 0.5,
    "xgboost_version": xgb.__version__,
    "dataset_sha256": digest.hexdigest(),
}
(MODEL_DIR / "xgboost_baseline_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

report = {
    "evaluation_partition": "validation",
    "threshold": 0.5,
    "training_rows": len(train_idx),
    "validation_rows": len(validation_idx),
    "model_parameters": model.get_params(),
    "precision": float(
        precision_score(y_validation, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_validation, predictions, zero_division=0)
    ),
    "f1": float(f1_score(y_validation, predictions, zero_division=0)),
    "average_precision": float(
        average_precision_score(y_validation, probabilities)
    ),
    "false_positive_rate": float(fp / (fp + tn)),
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
    "training_seconds": training_seconds,
    "validation_batch_prediction_seconds": prediction_seconds,
    "validation_rows_per_second": (
        len(y_validation) / prediction_seconds
    ),
    "model_size_bytes": model_path.stat().st_size,
    "per_attack_validation_recall": per_attack,
}

report_path = REPORT_DIR / "xgboost_validation.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nVALIDATION RESULTS")
print(json.dumps(report, indent=2))
print(f"\nModel saved: {model_path}")