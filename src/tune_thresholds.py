from pathlib import Path
import gc
import hashlib
import json
import time

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
OUT_DIR = REPORT_DIR / "tuning_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4096

dataset_path = DATA_DIR / "cleaned_flows.pkl"
split_report = json.loads((REPORT_DIR / "split_report.json").read_text())
features = json.loads((REPORT_DIR / "cleaning_report.json").read_text())["features"]

# Verify dataset
digest = hashlib.sha256()
with dataset_path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
dataset_hash = digest.hexdigest()
if dataset_hash != split_report["dataset_sha256"]:
    raise ValueError("Dataset changed after splitting")

print("Loading data...", flush=True)
data = pd.read_pickle(dataset_path)
with np.load(DATA_DIR / "split_indices.npz") as splits:
    train_idx = splits["train"]
    val_idx = splits["validation"]
    test_idx = splits["test"]

X_all = data[features].to_numpy(dtype=np.float32)
y_all = data["target"].to_numpy(dtype=np.uint8)
labels_all = data["original_label"].to_numpy()

X_val = X_all[val_idx]
y_val = y_all[val_idx]
val_labels = labels_all[val_idx]

X_test = X_all[test_idx]
y_test = y_all[test_idx]
test_labels = labels_all[test_idx]

del data, X_all, y_all, labels_all
gc.collect()

if not np.isfinite(X_val).all() or not np.isfinite(X_test).all():
    raise ValueError("Invalid values in arrays")

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

def predict_in_batches(estimator, values, scaler=None):
    probs = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(values))
        batch = values[start:end]
        if scaler is not None:
            batch = scaler.transform(batch, copy=True)
        probs[start:end] = estimator.predict_proba(batch)[:, 1]
    return probs

with threadpool_limits(limits=2):
    # Load models
    rf_bundle = joblib.load(MODEL_DIR / "random_forest_baseline.joblib")
    rf_model = rf_bundle["model"]
    rf_model.set_params(n_jobs=2)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_DIR / "xgboost_baseline.ubj")
    xgb_model.set_params(n_jobs=2, device="cpu")

    nn_bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
    nn_model = nn_bundle["model"]
    nn_scaler = nn_bundle["scaler"]

    models = {
        "rf": (rf_model, None),
        "xgb": (xgb_model, None),
        "nn": (nn_model, nn_scaler),
    }

    for name, (model, scaler) in models.items():
        print(f"\n=== Tuning {name.upper()} ===", flush=True)
        # Get validation probabilities
        val_probs = predict_in_batches(model, X_val, scaler=scaler)
        best_t = 0.5
        best_f1 = -1.0
        best_metrics = None
        for t in thresholds:
            preds = (val_probs >= t).astype(np.uint8)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
                best_metrics = {
                    "threshold": t,
                    "f1": float(f1),
                    "precision": float(precision_score(y_val, preds, zero_division=0)),
                    "recall": float(recall_score(y_val, preds, zero_division=0)),
                    "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                }
        print(f"Best on validation: t={best_t}, F1={best_f1:.6f}", flush=True)
        # Evaluate on test with best threshold
        test_probs = predict_in_batches(model, X_test, scaler=scaler)
        preds_test = (test_probs >= best_t).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_test, labels=[0, 1]).ravel()
        per_attack = {}
        for label in sorted(set(test_labels)):
            if label == "BENIGN":
                continue
            mask = test_labels == label
            count = int(mask.sum())
            detected = int(preds_test[mask].sum())
            per_attack[label] = {
                "rows": count,
                "detected": detected,
                "missed": count - detected,
                "recall": detected / count if count > 0 else 0.0,
            }
        report = {
            "model": name,
            "evaluation_partition": "test",
            "experiment": "threshold_tuning",
            "threshold": float(best_t),
            "rows": int(len(y_test)),
            "dataset_sha256": dataset_hash,
            "accuracy": float(accuracy_score(y_test, preds_test)),
            "precision": float(precision_score(y_test, preds_test, zero_division=0)),
            "recall": float(recall_score(y_test, preds_test, zero_division=0)),
            "f1": float(f1_score(y_test, preds_test, zero_division=0)),
            "average_precision": float(average_precision_score(y_test, test_probs)),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
            "per_attack_test_recall": per_attack,
            "validation_best": best_metrics,
        }
        out_path = OUT_DIR / f"{name}_threshold_tuned_test.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}", flush=True)
        print(f"Test: F1={report['f1']:.6f} Rec={report['recall']:.6f} Prec={report['precision']:.6f} FPR={report['false_positive_rate']:.6f}", flush=True)
        del val_probs, test_probs
        gc.collect()

print("\nDone threshold tuning.", flush=True)
