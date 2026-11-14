#!/usr/bin/env python3
"""ONNX correctness check.

Compare native sklearn MLP vs ONNX vs quantized-ONNX predictions on identical
records, using identical preprocessing (StandardScaler from the joblib bundle)
and the same decision threshold (validation-selected 0.7). ONNX models were
exported via skl2onnx and embed the scaler, so the ONNX path applies scaling
internally; the sklearn path applies the bundle's scaler explicitly.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "evaluation_final"
BATCH_SIZE = 4096
THRESHOLD = 0.7

features = json.loads((REPORT_DIR / "cleaning_report.json").read_text())["features"]


def load_test():
    import pandas as pd
    df = pd.read_pickle(DATA_DIR / "cleaned_flows.pkl")
    with np.load(DATA_DIR / "split_indices.npz") as s:
        test = df.iloc[s["test"]]
    return test[features].to_numpy(dtype=np.float32), test["target"].astype(int).to_numpy()


def run_onnx(path, X):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    probs = np.empty(len(X), dtype=np.float64)
    labels = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X))
        res = sess.run(None, {iname: X[start:end]})
        d = res[out_names.index("output_probability")]
        p1 = np.array([v.get(1, 0.0) for v in d], dtype=np.float64)
        probs[start:end] = p1
        labels[start:end] = res[out_names.index("output_label")].astype(np.int64)
    return probs, labels


def metrics(y, probs, thr):
    preds = (probs >= thr).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y)), "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probs)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def main():
    X, y = load_test()
    bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
    scaler = bundle["scaler"]
    native = bundle["model"]

    with threadpool_limits(limits=2):
        # Native sklearn path: explicit scaler
        Xs = scaler.transform(X, copy=True)
        native_proba = native.predict_proba(Xs)[:, 1]
        native_pred = (native_proba >= THRESHOLD).astype(np.uint8)

        onnx_proba, onnx_label = run_onnx(MODEL_DIR / "neural_network.onnx", X)
        onnxq_proba, onnxq_label = run_onnx(MODEL_DIR / "neural_network_quantized.onnx", X)

    rows = []
    for name, proba, pred in [
        ("native_sklearn", native_proba, native_pred),
        ("onnx", onnx_proba, (onnx_proba >= THRESHOLD).astype(np.uint8)),
        ("onnx_quantized", onnxq_proba, (onnxq_proba >= THRESHOLD).astype(np.uint8)),
    ]:
        m = metrics(y, proba, THRESHOLD)
        m["model"] = name
        m["threshold"] = THRESHOLD
        rows.append(m)
        print(f"{name}: F1={m['f1']:.6f} Rec={m['recall']:.6f} "
              f"Prec={m['precision']:.6f} FPR={m['fpr']:.6f}", flush=True)

    # Agreement vs native
    for name, proba, pred in [("onnx", onnx_proba, (onnx_proba >= THRESHOLD).astype(np.uint8)),
                              ("onnx_quantized", onnxq_proba, (onnxq_proba >= THRESHOLD).astype(np.uint8))]:
        agree = float(np.mean(pred == native_pred))
        maxdiff = float(np.max(np.abs(proba - native_proba)))
        meandiff = float(np.mean(np.abs(proba - native_proba)))
        print(f"{name}: agreement_with_native={agree:.6f} "
              f"mean_abs_score_diff={meandiff:.6f} max_abs_score_diff={maxdiff:.6f}", flush=True)
        rows.append({
            "model": name, "threshold": THRESHOLD, "rows": int(len(y)),
            "agreement_with_native": agree, "mean_abs_score_diff": meandiff,
            "max_abs_score_diff": maxdiff,
        })

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT_DIR / "onnx_agreement.csv", index=False)
    print("\nSaved:", OUT_DIR / "onnx_agreement.csv")


if __name__ == "__main__":
    sys.exit(main())
