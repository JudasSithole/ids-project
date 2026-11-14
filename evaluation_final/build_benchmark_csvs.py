#!/usr/bin/env python3
"""Flatten the five valid constrained benchmark JSONs into benchmark_runs.csv
and benchmark_summary.csv. These logs already contain five repeated runs under
documented settings (1 core affinity, 1 thread, batch 4096, validation partition,
503,881 rows); they are REUSED, not re-measured."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evaluation_final"
REPORTS = ROOT / "reports"

MODELS = ["rf", "xgb", "nn", "onnx", "onnxq"]
FILES = {
    "rf": "benchmark_rf_constrained_pi4_1core.json",
    "xgb": "benchmark_xgb_constrained_pi4_1core.json",
    "nn": "benchmark_nn_constrained_pi4_1core.json",
    "onnx": "benchmark_onnx_constrained_pi4_1core.json",
    "onnxq": "benchmark_onnxq_constrained_pi4_1core.json",
}

runs_rows = []
summary_rows = []
for m in MODELS:
    d = json.loads((REPORTS / FILES[m]).read_text())
    for r in d["runs"]:
        runs_rows.append({
            "model": m,
            "label": d["label"],
            "repeat": r["repeat"],
            "batch_size": d["batch_size"],
            "flows_evaluated": d["rows"],
            "cores_affinity": d["cores_affinity"],
            "threads": d["threads"],
            "wall_seconds": r["wall_seconds"],
            "cpu_seconds": r["cpu_seconds"],
            "cpu_percent_one_core_basis": r["average_cpu_percent_one_core_basis"],
            "rows_per_second": r["rows_per_second"],
            "median_batch_ms": r["median_batch_ms"],
            "p95_batch_ms": r["p95_batch_ms"],
            "timing_includes_load_preprocess": False,
        })
    walls = [r["wall_seconds"] for r in d["runs"]]
    rps = [r["rows_per_second"] for r in d["runs"]]
    summary_rows.append({
        "model": m,
        "label": d["label"],
        "partition": d["partition"],
        "batch_size": d["batch_size"],
        "flows_evaluated": d["rows"],
        "cores_affinity": d["cores_affinity"],
        "threads": d["threads"],
        "repeats": d["repeats"],
        "median_wall_seconds": float(np.median(walls)),
        "min_wall_seconds": float(np.min(walls)),
        "max_wall_seconds": float(np.max(walls)),
        "wall_range_seconds": float(np.max(walls) - np.min(walls)),
        "median_rows_per_second": float(np.median(rps)),
        "min_rows_per_second": float(np.min(rps)),
        "max_rows_per_second": float(np.max(rps)),
        "median_batch_ms": float(np.median([r["median_batch_ms"] for r in d["runs"]])),
        "sampled_peak_process_rss_mib": d["sampled_peak_process_rss_mib"],
        "sampled_rss_increase_mib": d["sampled_rss_increase_mib"],
        "timing_includes_load_preprocess": False,
        "cpu_percent_method": "psutil per-interval CPU seconds / wall seconds x 100, one-core basis",
        "rss_method": "sampled peak RSS of the benchmark process (data+libs+allocations); load excluded from timing",
    })

runs_df = pd.DataFrame(runs_rows)
summ_df = pd.DataFrame(summary_rows)
runs_df.to_csv(OUT / "benchmark_runs.csv", index=False)
summ_df.to_csv(OUT / "benchmark_summary.csv", index=False)
print(runs_df.to_string(index=False))
print()
print(summ_df.to_string(index=False))
print("\nSaved:", OUT / "benchmark_runs.csv", "and", OUT / "benchmark_summary.csv")
