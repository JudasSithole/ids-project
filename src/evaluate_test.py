from pathlib import Path
import gc
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
OUTPUT_DIR = REPORT_DIR / "initial_test"

BATCH_SIZE = 4096
THRESHOLD = 0.5

# Preserve any previous test evaluation.
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. Inspect its contents before rerunning."
    )

dataset_path = DATA_DIR / "cleaned_flows.pkl"
split_report = json.loads(
    (REPORT_DIR / "split_report.json").read_text()
)
features = json.loads(
    (REPORT_DIR / "cleaning_report.json").read_text()
)["features"]

print("Verifying dataset...", flush=True)
digest = hashlib.sha256()
with dataset_path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)

dataset_hash = digest.hexdigest()
if dataset_hash != split_report["dataset_sha256"]:
    raise ValueError("Dataset changed after splitting")

print("Loading test records...", flush=True)
data = pd.read_pickle(dataset_path)
with np.load(DATA_DIR / "split_indices.npz") as splits:
    test_indices = splits["test"]

test = data.iloc[test_indices]
X = test[features].to_numpy(dtype=np.float32)
y = test["target"].to_numpy(dtype=np.uint8)
original_labels = test["original_label"].to_numpy()

del data, test
gc.collect()

if not np.isfinite(X).all():
    raise ValueError("Invalid values after float32 conversion")

OUTPUT_DIR.mkdir(parents=True)
summaries = []

with threadpool_limits(limits=2):
    for name in ["rf", "xgb", "nn"]:
        print(f"\nEvaluating {name}...", flush=True)
        scaler = None
        bundle = None

        if name == "rf":
            model_path = MODEL_DIR / "random_forest_baseline.joblib"
            bundle = joblib.load(model_path)
            if bundle["features"] != features:
                raise ValueError("Random Forest feature order differs")
            if bundle["threshold"] != THRESHOLD:
                raise ValueError("Random Forest threshold differs")
            model = bundle["model"]
            model.set_params(n_jobs=2)

        elif name == "xgb":
            model_path = MODEL_DIR / "xgboost_baseline.ubj"
            metadata = json.loads(
                (MODEL_DIR / "xgboost_baseline_metadata.json").read_text()
            )
            if metadata["features"] != features:
                raise ValueError("XGBoost feature order differs")
            if metadata["threshold"] != THRESHOLD:
                raise ValueError("XGBoost threshold differs")
            if metadata["dataset_sha256"] != dataset_hash:
                raise ValueError("XGBoost dataset fingerprint differs")
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            model.set_params(n_jobs=2, device="cpu")

        else:
            model_path = MODEL_DIR / "neural_network_baseline.joblib"
            bundle = joblib.load(model_path)
            if bundle["features"] != features:
                raise ValueError("NN feature order differs")
            if bundle["threshold"] != THRESHOLD:
                raise ValueError("NN threshold differs")
            if bundle["dataset_sha256"] != dataset_hash:
                raise ValueError("NN dataset fingerprint differs")
            model = bundle["model"]
            scaler = bundle["scaler"]

        if list(model.classes_) != [0, 1]:
            raise ValueError(f"Unexpected class order for {name}")

        probabilities = np.empty(len(y), dtype=np.float64)

        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))
            batch = X[start:end]

            if scaler is not None:
                batch = scaler.transform(batch, copy=True)

            probabilities[start:end] = model.predict_proba(batch)[:, 1]

        predictions = (probabilities >= THRESHOLD).astype(np.uint8)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        per_attack = {}
        for label in sorted(set(original_labels)):
            if label == "BENIGN":
                continue
            mask = original_labels == label
            count = int(mask.sum())
            detected = int(predictions[mask].sum())
            per_attack[label] = {
                "rows": count,
                "detected": detected,
                "missed": count - detected,
                "recall": detected / count,
            }

        model_digest = hashlib.sha256()
        with model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                model_digest.update(block)

        report = {
            "model": name,
            "evaluation_partition": "test",
            "experiment": "initial_baseline",
            "threshold": THRESHOLD,
            "rows": len(y),
            "dataset_sha256": dataset_hash,
            "model_sha256": model_digest.hexdigest(),
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(
                precision_score(y, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "average_precision": float(
                average_precision_score(y, probabilities)
            ),
            "false_positive_rate": float(fp / (fp + tn)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "per_attack_test_recall": per_attack,
        }

        (OUTPUT_DIR / f"{name}_test.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        # Save predictions for traceable error analysis.
        np.savez_compressed(
            OUTPUT_DIR / f"{name}_test_predictions.npz",
            cleaned_dataset_row_indices=test_indices,
            true_labels=y,
            attack_probabilities=probabilities,
            predicted_labels=predictions,
        )

        summaries.append({
            "model": name,
            "precision": report["precision"],
            "recall": report["recall"],
            "f1": report["f1"],
            "false_positive_rate": report["false_positive_rate"],
            "false_positives": int(fp),
            "missed_attacks": int(fn),
        })

        print(json.dumps(report, indent=2), flush=True)
        del model, bundle, scaler, probabilities, predictions
        gc.collect()

pd.DataFrame(summaries).to_csv(
    OUTPUT_DIR / "test_comparison.csv", index=False
)

print("\nTEST COMPARISON")
print(pd.DataFrame(summaries).to_string(index=False))
print(f"\nResults saved in: {OUTPUT_DIR}")