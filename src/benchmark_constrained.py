#!/usr/bin/env python3
"""
Constrained environment benchmark for resource-constrained IDS evaluation.
Simulates low-spec hardware (e.g., Raspberry Pi) using core affinity and resource limits.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
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
REPORT_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Benchmark models under constrained conditions")
parser.add_argument("model", choices=["rf", "xgb", "nn", "onnx", "onnxq"])
parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores to use (default: 1)")
parser.add_argument("--threads", type=int, default=1, help="Threadpool limit (default: 1)")
parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for inference")
parser.add_argument("--repeats", type=int, default=5, help="Number of benchmark repeats")
parser.add_argument("--label", type=str, default=None, help="Label for this run (e.g., 'pi4_1core')")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
REPEATS = args.repeats

# Set environment variables for thread limiting
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

dataset_path = DATA_DIR / "cleaned_flows.pkl"
split_report = json.loads((REPORT_DIR / "split_report.json").read_text())
features = json.loads((REPORT_DIR / "cleaning_report.json").read_text())["features"]

digest = hashlib.sha256()
with dataset_path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != split_report["dataset_sha256"]:
    raise ValueError("Dataset changed after splitting")

print(f"Preparing {args.model} benchmark under constrained conditions...", flush=True)
print(f"  Cores: {args.cores}, Threads: {args.threads}, Batch size: {BATCH_SIZE}", flush=True)

data = pd.read_pickle(dataset_path)
with np.load(DATA_DIR / "split_indices.npz") as splits:
    indices = splits["validation"]

X = data.iloc[indices][features].to_numpy(dtype=np.float32)
del data
gc.collect()
if not np.isfinite(X).all():
    raise ValueError("Invalid feature values")

scaler = None
model = None
session = None

if args.model == "rf":
    bundle = joblib.load(MODEL_DIR / "random_forest_baseline.joblib")
    assert bundle["features"] == features
    model = bundle["model"]
    model.set_params(n_jobs=args.threads)
elif args.model == "xgb":
    metadata = json.loads((MODEL_DIR / "xgboost_baseline_metadata.json").read_text())
    assert metadata["features"] == features
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "xgboost_baseline.ubj")
    model.set_params(n_jobs=args.threads, device="cpu")
elif args.model == "nn":
    bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
    assert bundle["features"] == features
    model = bundle["model"]
    scaler = bundle["scaler"]
elif args.model in ["onnx", "onnxq"]:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Run: pip install onnxruntime", flush=True)
        raise
    if args.model == "onnxq":
        onnx_path = MODEL_DIR / "neural_network_quantized.onnx"
    else:
        onnx_path = MODEL_DIR / "neural_network.onnx"
    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found. Run src/export_onnx.py first.", flush=True)
        raise FileNotFoundError(f"{onnx_path} not found")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.threads
    sess_options.inter_op_num_threads = args.threads
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Loaded ONNX model from {onnx_path}", flush=True)

def predict_batch(batch):
    if scaler is not None and args.model == "nn":
        batch_scaled = scaler.transform(batch, copy=True)
        return model.predict_proba(batch_scaled)[:, 1]
    elif args.model in ["onnx", "onnxq"] and session is not None:
        # ONNX expects float32
        inputs = {input_name: batch.astype(np.float32)}
        outputs = session.run(None, inputs)
        # Assuming output shape varies
        probs = outputs[0]
        if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        elif isinstance(probs, np.ndarray):
            # Could be probabilities for positive class or just scores
            if probs.ndim == 1:
                return probs
            return probs[:, -1]
        return np.array(probs)
    elif model is not None:
        return model.predict_proba(batch)[:, 1]
    raise ValueError("No model loaded")

process = psutil.Process()
# Pin to specific cores if possible
try:
    if args.cores >= 1:
        process.cpu_affinity([0] if args.cores == 1 else list(range(args.cores)))
        print(f"Pinned process to CPU core(s): {process.cpu_affinity()}", flush=True)
except Exception as e:
    print(f"Warning: Could not set CPU affinity: {e}", flush=True)

stop = threading.Event()
memory_samples = []

def monitor_memory():
    while not stop.is_set():
        try:
            memory_samples.append(process.memory_info().rss)
        except:
            pass
        stop.wait(0.01)

def cpu_seconds():
    try:
        values = process.cpu_times()
        return values.user + values.system
    except:
        return 0.0

runs = []
with threadpool_limits(limits=args.threads):
    # Warm up
    print("Warming up...", flush=True)
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
                    100 * used_cpu_seconds / wall_seconds if wall_seconds > 0 else 0
                ),
                "rows_per_second": len(X) / wall_seconds if wall_seconds > 0 else 0,
                "median_batch_ms": float(np.median(batch_seconds) * 1000),
                "p95_batch_ms": float(np.percentile(batch_seconds, 95) * 1000),
                "predicted_attack_rows": positive_count,
            })
            print(
                f"Run {repeat}: {wall_seconds:.3f}s, "
                f"{len(X) / wall_seconds:,.0f} rows/s, "
                f"CPU: {100 * used_cpu_seconds / wall_seconds:.1f}%",
                flush=True,
            )
    finally:
        stop.set()
        monitor.join()
        memory_samples.append(process.memory_info().rss)

if runs:
    assert len({run["predicted_attack_rows"] for run in runs}) == 1

median_wall = float(np.median([run["wall_seconds"] for run in runs])) if runs else 0
report = {
    "model": args.model,
    "label": args.label or f"{args.cores}core_{args.threads}thread",
    "partition": "validation",
    "rows": len(X),
    "batch_size": BATCH_SIZE,
    "cores_affinity": args.cores,
    "threads": args.threads,
    "repeats": REPEATS,
    "dataset_sha256": digest.hexdigest(),
    "median_wall_seconds": median_wall,
    "rows_per_second_at_median_time": len(X) / median_wall if median_wall > 0 else 0,
    "process_rss_before_measured_runs_mib": baseline_rss / 1024**2,
    "sampled_peak_process_rss_mib": max(memory_samples) / 1024**2,
    "sampled_rss_increase_mib": (max(memory_samples) - baseline_rss) / 1024**2,
    "runs": runs,
    "environment": {
        "system": os.uname().sysname,
        "machine": os.uname().machine,
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
    },
    "notes": [
        f"Constrained benchmark - simulated low-spec environment",
        f"CPU affinity set to {args.cores} core(s), threadpool limited to {args.threads}",
        "NN/ONNX scaling included where applicable.",
        "Data loading, model loading and warmup are excluded from timing.",
        "RSS includes data, libraries and retained allocations.",
    ],
}

suffix = f"constrained_{args.label}" if args.label else f"constrained_{args.cores}core"
output = REPORT_DIR / f"benchmark_{args.model}_{suffix}.json"
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n" + "="*60)
print("CONSTRAINED BENCHMARK RESULTS")
print("="*60)
print(json.dumps(report, indent=2))
print(f"\nResults saved to: {output}")
