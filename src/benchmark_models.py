from pathlib import Path
import argparse
import gc
import hashlib
import json
import threading
import time

import joblib
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["rf", "xgb", "nn"])
args = parser.parse_args()

BATCH_SIZE = 4096
REPEATS = 5
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

print(f"Preparing {args.model} benchmark...", flush=True)
data = pd.read_pickle(dataset_path)
with np.load(DATA_DIR / "split_indices.npz") as splits:
    indices = splits["validation"]

X = data.iloc[indices][features].to_numpy(dtype=np.float32)
del data
gc.collect()
if not np.isfinite(X).all():
    raise ValueError("Invalid feature values")

scaler = None
if args.model == "rf":
    bundle = joblib.load(MODEL_DIR / "random_forest_baseline.joblib")
    assert bundle["features"] == features
    model = bundle["model"]
    model.set_params(n_jobs=2)
elif args.model == "xgb":
    metadata = json.loads(
        (MODEL_DIR / "xgboost_baseline_metadata.json").read_text()
    )
    assert metadata["features"] == features
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "xgboost_baseline.ubj")
    model.set_params(n_jobs=2, device="cpu")
else:
    bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
    assert bundle["features"] == features
    model = bundle["model"]
    scaler = bundle["scaler"]

def predict_batch(batch):
    if scaler is not None:
        # Never change the original validation features.
        batch = scaler.transform(batch, copy=True)
    return model.predict_proba(batch)[:, 1]

process = psutil.Process()
stop = threading.Event()
memory_samples = []

def monitor_memory():
    while not stop.is_set():
        memory_samples.append(process.memory_info().rss)
        stop.wait(0.01)

def cpu_seconds():
    values = process.cpu_times()
    return values.user + values.system

runs = []
with threadpool_limits(limits=2):
    # Warm up every batch before measuring.
    for start in range(0, len(X), BATCH_SIZE):
        predict_batch(X[start:start + BATCH_SIZE])

    baseline_rss = process.memory_info().rss
    memory_samples.append(baseline_rss)
    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()

    try:
        for repeat in range(1, REPEATS + 1):
            batch_seconds = []
            positive_count = 0
            cpu_start = cpu_seconds()
            wall_start = time.perf_counter()

            for start in range(0, len(X), BATCH_SIZE):
                batch = X[start:start + BATCH_SIZE]
                started = time.perf_counter()
                probabilities = predict_batch(batch)
                batch_seconds.append(time.perf_counter() - started)
                positive_count += int((probabilities >= 0.5).sum())

            wall_seconds = time.perf_counter() - wall_start
            used_cpu_seconds = cpu_seconds() - cpu_start

            runs.append({
                "repeat": repeat,
                "wall_seconds": wall_seconds,
                "cpu_seconds": used_cpu_seconds,
                "average_cpu_percent_one_core_basis": (
                    100 * used_cpu_seconds / wall_seconds
                ),
                "rows_per_second": len(X) / wall_seconds,
                "median_batch_ms": float(
                    np.median(batch_seconds) * 1000
                ),
                "p95_batch_ms": float(
                    np.percentile(batch_seconds, 95) * 1000
                ),
                "predicted_attack_rows": positive_count,
            })
            print(
                f"Run {repeat}: {wall_seconds:.3f}s, "
                f"{len(X) / wall_seconds:,.0f} rows/s",
                flush=True,
            )
    finally:
        stop.set()
        monitor.join()
        memory_samples.append(process.memory_info().rss)

assert len({run["predicted_attack_rows"] for run in runs}) == 1

median_wall = float(np.median([run["wall_seconds"] for run in runs]))
report = {
    "model": args.model,
    "partition": "validation",
    "rows": len(X),
    "batch_size": BATCH_SIZE,
    "workers": 2,
    "repeats": REPEATS,
    "dataset_sha256": digest.hexdigest(),
    "median_wall_seconds": median_wall,
    "rows_per_second_at_median_time": len(X) / median_wall,
    "process_rss_before_measured_runs_mib": baseline_rss / 1024**2,
    "sampled_peak_process_rss_mib": max(memory_samples) / 1024**2,
    "sampled_rss_increase_mib": (
        max(memory_samples) - baseline_rss
    ) / 1024**2,
    "runs": runs,
    "notes": [
        "NN scaling is included.",
        "Data loading, model loading and warmup are excluded from timing.",
        "RSS includes data, libraries and retained allocations, not just model.",
        "Memory sampled every 10ms; brief peaks may be missed.",
        "Batch timing is not single-flow or packet-to-alert latency.",
        "CPU percent uses one core as 100%; two busy cores can approach 200%.",
    ],
}

output = REPORT_DIR / f"benchmark_{args.model}.json"
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nBENCHMARK RESULTS")
print(json.dumps(report, indent=2))