from pathlib import Path
import copy
import gc
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 20
PATIENCE = 4

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
    raise ValueError("Invalid values after float32 conversion")

X_train = X[train_idx]
y_train = y[train_idx]
X_validation = X[validation_idx]
y_validation = y[validation_idx]
validation_labels = labels[validation_idx]
del X, y, labels
gc.collect()

scaler = StandardScaler(copy=False)
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=BATCH_SIZE,
    learning_rate_init=0.001,
    shuffle=False,  # We shuffle explicitly each epoch.
    early_stopping=False,
    random_state=SEED,
)

rng = np.random.default_rng(SEED)
history = []
best_model = None
best_f1 = -1.0
best_epoch = 0
stale_epochs = 0
optimizer_seconds = 0.0

def predict_in_batches(estimator, values):
    return np.concatenate([
        estimator.predict_proba(values[start:start + BATCH_SIZE])[:, 1]
        for start in range(0, len(values), BATCH_SIZE)
    ])

# Limit numerical-library workers to two for this initial experiment.
with threadpool_limits(limits=2):
    started = time.perf_counter()
    scaler.fit(X_train)  # Training data only.
    X_train = scaler.transform(X_train)
    X_validation = scaler.transform(X_validation)
    preprocessing_seconds = time.perf_counter() - started

    # Normalize class weights to have an average weight of one.
    counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * counts)

    training_started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"Training epoch {epoch}/{MAX_EPOCHS}...", flush=True)
        order = rng.permutation(len(y_train))
        started = time.perf_counter()

        for start in range(0, len(order), BATCH_SIZE):
            selected = order[start:start + BATCH_SIZE]
            model.partial_fit(
                X_train[selected],
                y_train[selected],
                classes=np.array([0, 1], dtype=np.uint8),
                sample_weight=class_weights[y_train[selected]],
            )

        optimizer_seconds += time.perf_counter() - started
        probabilities = predict_in_batches(model, X_validation)
        predictions = (probabilities >= 0.5).astype(np.uint8)
        score = float(f1_score(y_validation, predictions, zero_division=0))

        history.append({"epoch": epoch, "validation_f1": score})
        print(f"Validation F1: {score:.6f}", flush=True)

        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print("Stopping: no validation F1 improvement for 4 epochs.")
            break

    training_wall_seconds = time.perf_counter() - training_started
    model = best_model

    started = time.perf_counter()
    probabilities = predict_in_batches(model, X_validation)
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

model_path = MODEL_DIR / "neural_network_baseline.joblib"
joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": features,
    "threshold": 0.5,
    "dataset_sha256": digest.hexdigest(),
}, model_path, compress=3)

report = {
    "evaluation_partition": "validation",
    "architecture": [32, 16],
    "threshold": 0.5,
    "training_rows": len(train_idx),
    "validation_rows": len(validation_idx),
    "best_epoch": best_epoch,
    "epochs_run": len(history),
    "epoch_history": history,
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
    "preprocessing_seconds": preprocessing_seconds,
    "optimizer_seconds": optimizer_seconds,
    "training_wall_seconds_including_validation": training_wall_seconds,
    "validation_prediction_seconds_excluding_scaling": prediction_seconds,
    "validation_rows_per_second": len(y_validation) / prediction_seconds,
    "compressed_model_and_scaler_size_bytes": model_path.stat().st_size,
    "per_attack_validation_recall": per_attack,
    "model_parameters": model.get_params(),
}

report_path = REPORT_DIR / "neural_network_validation.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nVALIDATION RESULTS")
print(json.dumps(report, indent=2))
print(f"\nModel and scaler saved: {model_path}")