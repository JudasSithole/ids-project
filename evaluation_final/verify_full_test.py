#!/usr/bin/env python3
"""Recompute full-test-set detection metrics for the three tuned models.

Thresholds are the validation-selected values recorded in
reports/tuning_test/*_threshold_tuned_test.json (RF=0.6, XGB=0.7, MLP=0.7).
Reproduces Table I of the draft report from saved models + saved split indices.
Thresholds are NOT selected from test labels.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "evaluation_final"
BATCH_SIZE = 4096

THRESHOLDS = {"rf": 0.6, "xgb": 0.7, "nn": 0.7}

dataset_path = DATA_DIR / "cleaned_flows.pkl"
split_report = json.loads((REPORT_DIR / "split_report.json").read_text())
features = json.loads((REPORT_DIR / "cleaning_report.json").read_text())["features"]

digest = hashlib.sha256()
with dataset_path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
dataset_hash = digest.hexdigest()
assert dataset_hash == split_report["dataset_sha256"], "dataset changed"

data = pd.read_pickle(dataset_path)
with np.load(DATA_DIR / "split_indices.npz") as splits:
    test_idx = splits["test"]

X = data.iloc[test_idx][features].to_numpy(dtype=np.float32)
y = data.iloc[test_idx]["target"].to_numpy(dtype=np.uint8)
orig_labels = data.iloc[test_idx]["original_label"].to_numpy()
del data

rows = {k: [] for k in [
    "model", "threshold", "rows", "accuracy", "precision", "recall", "f1",
    "pr_auc", "fpr", "tn", "fp", "fn", "tp"]}


def predict_in_batches(model, values, scaler=None):
    probs = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(values))
        batch = values[start:end]
        if scaler is not None:
            batch = scaler.transform(batch, copy=True)
        probs[start:end] = model.predict_proba(batch)[:, 1]
    return probs


with threadpool_limits(limits=2):
    for name, thr in THRESHOLDS.items():
        if name == "rf":
            bundle = joblib.load(MODEL_DIR / "random_forest_baseline.joblib")
            assert bundle["features"] == features
            model = bundle["model"]
            model.set_params(n_jobs=2)
            scaler = None
        elif name == "xgb":
            md = json.loads((MODEL_DIR / "xgboost_baseline_metadata.json").read_text())
            assert md["features"] == features
            model = xgb.XGBClassifier()
            model.load_model(MODEL_DIR / "xgboost_baseline.ubj")
            model.set_params(n_jobs=2, device="cpu")
            scaler = None
        else:
            bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
            assert bundle["features"] == features
            model = bundle["model"]
            scaler = bundle["scaler"]

        probs = predict_in_batches(model, X, scaler=scaler)
        preds = (probs >= thr).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        rows["model"].append(name)
        rows["threshold"].append(thr)
        rows["rows"].append(int(len(y)))
        rows["accuracy"].append(float(accuracy_score(y, preds)))
        rows["precision"].append(float(precision_score(y, preds, zero_division=0)))
        rows["recall"].append(float(recall_score(y, preds, zero_division=0)))
        rows["f1"].append(float(f1_score(y, preds, zero_division=0)))
        rows["pr_auc"].append(float(average_precision_score(y, probs)))
        rows["fpr"].append(float(fp / (fp + tn)))
        rows["tn"].append(int(tn))
        rows["fp"].append(int(fp))
        rows["fn"].append(int(fn))
        rows["tp"].append(int(tp))
        print(f"{name} thr={thr}: F1={rows['f1'][-1]:.6f} "
              f"Rec={rows['recall'][-1]:.6f} Prec={rows['precision'][-1]:.6f} "
              f"FPR={rows['fpr'][-1]:.6f} rows={rows['rows'][-1]}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "detection_metrics_full_test.csv", index=False)
print("\nSaved:", OUT_DIR / "detection_metrics_full_test.csv")
print(df.to_string(index=False))
