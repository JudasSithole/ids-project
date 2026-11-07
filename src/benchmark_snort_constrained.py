#!/usr/bin/env python3
"""Benchmark Snort offline detection on a PCAP under emulated low-spec (1 core) constraints.

Measures wall time, throughput, and sampled CPU/RSS while running Snort -r on a PCAP.
Outputs JSON comparable in spirit to reports/benchmark_*_constrained_*.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]


def sample_process(proc: psutil.Process, stop_event: dict, samples: list, interval: float):
    # simple polling loop (called from main thread logic via while)
    try:
        while proc.is_running() and not stop_event.get("stop"):
            try:
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_info().rss / (1024 * 1024)
                # include children
                for ch in proc.children(recursive=True):
                    try:
                        cpu += ch.cpu_percent(interval=None)
                        mem += ch.memory_info().rss / (1024 * 1024)
                    except Exception:
                        pass
                samples.append({"ts": time.time(), "cpu_percent_sum": cpu, "rss_mib": mem})
            except Exception:
                pass
            time.sleep(interval)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--config", default=str(ROOT / "snort_friday_alert.lua"))
    ap.add_argument("--log-dir", default=str(ROOT / "snort_out" / "resource_bench"))
    ap.add_argument("--snort", default="/usr/local/bin/snort")
    ap.add_argument("--cores", default="0", help="CPU core affinity, e.g. 0")
    ap.add_argument("--sample-interval", type=float, default=0.5)
    ap.add_argument("--label", default="pi4_1core_emulated")
    ap.add_argument("-o", "--output", default=str(ROOT / "reports" / "benchmark_snort_constrained.json"))
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    pcap = Path(args.pcap)
    if not pcap.exists():
        raise SystemExit(f"PCAP not found: {pcap}")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "taskset", "-c", args.cores,
        args.snort,
        "-c", args.config,
        "-r", str(pcap),
        "-A", "alert_fast",
        "-l", str(log_dir),
        "-q",
    ] + (args.extra_args or [])

    print("Running:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ps = psutil.Process(proc.pid)
    # prime cpu_percent
    try:
        ps.cpu_percent(interval=None)
        for ch in ps.children(recursive=True):
            ch.cpu_percent(interval=None)
    except Exception:
        pass
    samples = []
    stop = {"stop": False}
    # poll in-loop
    while True:
        ret = proc.poll()
        try:
            if ps.is_running():
                cpu = ps.cpu_percent(interval=None)
                mem = ps.memory_info().rss / (1024 * 1024)
                for ch in ps.children(recursive=True):
                    try:
                        cpu += ch.cpu_percent(interval=None)
                        mem += ch.memory_info().rss / (1024 * 1024)
                    except Exception:
                        pass
                samples.append({"ts": time.time() - t0, "cpu_percent_sum": cpu, "rss_mib": mem})
        except Exception:
            pass
        if ret is not None:
            break
        time.sleep(args.sample_interval)

    wall = time.time() - t0
    stdout, stderr = proc.communicate()
    peak_rss = max((s["rss_mib"] for s in samples), default=0.0)
    mean_cpu = sum((s["cpu_percent_sum"] for s in samples), 0.0) / len(samples) if samples else 0.0
    pcap_bytes = pcap.stat().st_size

    result = {
        "model": "snort3_community",
        "label": args.label,
        "pcap": str(pcap),
        "pcap_bytes": pcap_bytes,
        "config": args.config,
        "log_dir": str(log_dir),
        "command": cmd,
        "returncode": proc.returncode,
        "wall_seconds": wall,
        "pcap_mb_per_second": (pcap_bytes / (1024 * 1024)) / wall if wall else None,
        "sampled_peak_process_rss_mib": peak_rss,
        "sampled_mean_cpu_percent_sum": mean_cpu,
        "sample_count": len(samples),
        "sample_interval_seconds": args.sample_interval,
        "cores_affinity": args.cores,
        "snort_version": subprocess.getoutput(f"{args.snort} -V | head -2"),
        "stderr_tail": "\n".join((stderr or "").splitlines()[-40:]),
        "stdout_tail": "\n".join((stdout or "").splitlines()[-40:]),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
        },
        "notes": [
            "Offline PCAP read (-r) with -q; throughput is PCAP MB/s, not live capture throughput.",
            "CPU percent is psutil's per-interval sum over process+children; on multi-core hosts values can exceed 100 but affinity was constrained via taskset.",
            "Emulated low-spec: intended to approximate constrained single-core behavior; no physical Raspberry Pi used.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["returncode", "wall_seconds", "pcap_mb_per_second", "sampled_peak_process_rss_mib", "sampled_mean_cpu_percent_sum"]}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
