#!/usr/bin/env python3
"""Recompute Friday test-subset ML metrics at BOTH thresholds.

The saved reports/friday_ml_tuned_subset.json reports RF at tau=0.35, but the
validation-selected threshold (from src/tune_thresholds.py max-F1 sweep, recorded
in reports/tuning_test/rf_threshold_tuned_test.json) is 0.6.

This script reproduces the Friday subset EXACTLY as
src/correlate_snort_friday_fast.py does (test split + source_file contains
"Friday"), applies the saved feature order, and evaluates each model at both
thresholds. No test labels are used to choose any threshold -- both are
pre-specified.
"""
import json
import sys
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "evaluation_final"
BATCH_SIZE = 4096

# Validation-selected thresholds (source of truth: tuning_test JSONs)
VAL_THR = {"rf": 0.6, "xgb": 0.7, "nn": 0.7}
# Thresholds as they appear in reports/friday_ml_tuned_subset.json
REP_THR = {"rf": 0.35, "xgb": 0.7, "nn": 0.7}


def load_friday_test():
    """Reproduce correlate_snort_friday_fast.load_friday_test exactly."""
    df = pd.read_pickle(DATA_DIR / "cleaned_flows.pkl")
    with np.load(DATA_DIR / "split_indices.npz") as splits:
        test = df.iloc[splits["test"]].copy()
    fri = test[test["source_file"].str.contains("Friday", case=False, na=False)].copy()
    return fri


def predict_in_batches(model, values, scaler=None):
    probs = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(values))
        batch = values[start:end]
        if scaler is not None:
            batch = scaler.transform(batch, copy=True)
        probs[start:end] = model.predict_proba(batch)[:, 1]
    return probs


def make_models(features):
    out = {}
    bundle = joblib.load(MODEL_DIR / "random_forest_baseline.joblib")
    assert bundle["features"] == features
    bundle["model"].set_params(n_jobs=2)
    out["rf"] = (bundle["model"], None)

    md = json.loads((MODEL_DIR / "xgboost_baseline_metadata.json").read_text())
    assert md["features"] == features
    m = xgb.XGBClassifier()
    m.load_model(MODEL_DIR / "xgboost_baseline.ubj")
    m.set_params(n_jobs=2, device="cpu")
    out["xgb"] = (m, None)

    bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
    assert bundle["features"] == features
    out["nn"] = (bundle["model"], bundle["scaler"])
    return out


def metrics(y, probs, thr):
    preds = (probs >= thr).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "rows": int(len(y)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probs)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def main():
    features = json.loads((REPORT_DIR / "cleaning_report.json").read_text())["features"]
    fri = load_friday_test()
    y = fri["target"].astype(int).to_numpy()
    X = fri[features].to_numpy(dtype=np.float32)
    print(f"Friday subset: {len(fri)} rows", flush=True)

    models = make_models(features)
    out = {}
    with threadpool_limits(limits=2):
        for name, (model, scaler) in models.items():
            probs = predict_in_batches(model, X, scaler=scaler)
            row = {"model": name,
                   "validation_selected_threshold": VAL_THR[name],
                   "reported_friday_threshold": REP_THR[name]}
            for label, thr in (("validation_selected", VAL_THR[name]),
                               ("reported", REP_THR[name])):
                for k, v in metrics(y, probs, thr).items():
                    row[f"{label}_{k}"] = v
            out[name] = row
            print(f"{name}: val-thr={VAL_THR[name]} F1={row['validation_selected_f1']:.6f} "
                  f"| rep-thr={REP_THR[name]} F1={row['reported_f1']:.6f}", flush=True)

    pd.DataFrame([out[k] for k in out]).to_csv(
        OUT_DIR / "detection_metrics_friday_both_thresholds.csv", index=False)
    (OUT_DIR / "friday_recompute.json").write_text(json.dumps(out, indent=2))
    print("Saved:", OUT_DIR / "detection_metrics_friday_both_thresholds.csv")


if __name__ == "__main__":
    sys.exit(main())
