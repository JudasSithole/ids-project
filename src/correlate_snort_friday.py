#!/usr/bin/env python3
"""
Correlate parsed Snort alerts to Friday CICIDS2017 flows in the test split.
Outputs detection metrics for Snort on the same Friday subset used for ML comparison.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SHARE = Path("/media/sf_OdinShare")
LABELLED_DIR = SHARE / "GeneratedLabelledFlows" / "TrafficLabelling"

FRIDAY_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


def norm_proto(p):
    if p is None:
        return None
    s = str(p).strip().upper()
    if s in {"TCP", "6", "TCP6"}:
        return 6
    if s in {"UDP", "17"}:
        return 17
    if s in {"ICMP", "1"}:
        return 1
    try:
        return int(float(s))
    except Exception:
        return None


def parse_flow_ts(ts):
    if pd.isna(ts):
        return None
    s = str(ts).strip()
    for fmt in ("%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, dayfirst=True).to_pydatetime()
    except Exception:
        return None


def parse_alert_ts(ts_str, year=2017):
    # Snort fast: MM/DD-HH:MM:SS.ffffff
    try:
        dt = datetime.strptime(f"{year}/{ts_str}", "%Y/%m/%d-%H:%M:%S.%f")
        return dt
    except Exception:
        return None


def load_labelled_friday_metadata():
    frames = []
    for name in FRIDAY_FILES:
        path = LABELLED_DIR / name
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        # source_row is 1-based data row index (header excluded)
        df["source_file"] = name
        df["source_row"] = np.arange(1, len(df) + 1, dtype=np.int64)
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    # normalize columns
    rename = {}
    for c in meta.columns:
        cl = c.lower()
        if cl == "source ip":
            rename[c] = "src_ip"
        elif cl == "source port":
            rename[c] = "src_port"
        elif cl == "destination ip":
            rename[c] = "dst_ip"
        elif cl == "destination port":
            rename[c] = "dst_port"
        elif cl == "protocol":
            rename[c] = "protocol"
        elif cl == "timestamp":
            rename[c] = "timestamp"
        elif "flow duration" in cl:
            rename[c] = "flow_duration"
    meta = meta.rename(columns=rename)
    for col in ["src_port", "dst_port"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    if "protocol" in meta.columns:
        meta["proto_num"] = meta["protocol"].apply(norm_proto)
    if "timestamp" in meta.columns:
        meta["start_dt"] = meta["timestamp"].apply(parse_flow_ts)
    if "flow_duration" in meta.columns:
        # CICFlowMeter duration is typically microseconds
        meta["end_dt"] = meta.apply(
            lambda r: r["start_dt"] + timedelta(microseconds=float(r["flow_duration"]))
            if pd.notna(r.get("start_dt")) and pd.notna(r.get("flow_duration"))
            else None,
            axis=1,
        )
    return meta


def build_test_friday_df():
    df = pickle.load(open(DATA_PROCESSED / "cleaned_flows.pkl", "rb"))
    test_idx = np.load(DATA_PROCESSED / "split_indices.npz")["test"]
    test = df.iloc[test_idx].copy()
    test["test_pos"] = np.arange(len(test))
    fri = test[test["source_file"].str.contains("Friday", case=False, na=False)].copy()
    return fri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="reports/snort_alerts_parsed.json")
    ap.add_argument("--delta-seconds", type=float, default=2.0, help="Time window tolerance around flow start/end")
    ap.add_argument(
        "--alert-offset-seconds",
        type=float,
        default=0.0,
        help="Seconds subtracted from alert timestamp after parsing (alert_dt -= offset). Use to align Snort/pcap time to CICFlowMeter flow timestamps.",
    )
    ap.add_argument(
        "--auto-offset",
        action="store_true",
        help="Estimate offset from tuple-matched TCP/UDP alerts before correlation (overrides --alert-offset-seconds when successful).",
    )
    ap.add_argument("-o", "--output", default="reports/snort_friday_evaluation.json")
    args = ap.parse_args()

    alerts_path = ROOT / args.alerts
    payload = json.loads(alerts_path.read_text(encoding="utf-8"))
    alerts = payload["alerts"] if isinstance(payload, dict) and "alerts" in payload else payload
    print(f"Loaded {len(alerts)} parsed alerts from {alerts_path}")

    fri_test = build_test_friday_df()
    meta = load_labelled_friday_metadata()
    needed = meta[["source_file", "source_row", "src_ip", "src_port", "dst_ip", "dst_port", "proto_num", "start_dt", "end_dt"]].copy()
    merged = fri_test.merge(needed, on=["source_file", "source_row"], how="left")
    missing = merged["start_dt"].isna().sum()
    print(f"Friday test flows: {len(merged)}, missing metadata: {missing}")

    offset_seconds = float(args.alert_offset_seconds)
    offset_est = None
    if args.auto_offset:
        flow_keys = defaultdict(list)
        for r in merged.itertuples(index=False):
            sp = int(r.src_port) if pd.notna(r.src_port) else None
            dp = int(r.dst_port) if pd.notna(r.dst_port) else None
            for k in [(r.src_ip, sp, r.dst_ip, dp), (r.dst_ip, dp, r.src_ip, sp)]:
                if pd.notna(r.start_dt):
                    flow_keys[k].append(r.start_dt)
        deltas = []
        checked = 0
        for a in alerts:
            if a.get("proto") not in ("TCP", "UDP") or not a.get("src_port"):
                continue
            k = (a.get("src_ip"), a.get("src_port"), a.get("dst_ip"), a.get("dst_port"))
            if k not in flow_keys:
                continue
            adt = parse_alert_ts(a.get("timestamp", ""), year=2017)
            if adt is None:
                continue
            for fdt in flow_keys[k]:
                deltas.append((adt - fdt).total_seconds())
            checked += 1
            if checked >= 20000:
                break
        if deltas:
            offset_est = float(pd.Series(deltas).median())
            offset_seconds = offset_est
            print(f"Auto-estimated alert offset: {offset_est:.3f}s (checked {checked} tuple-matched alerts, {len(deltas)} deltas)")
        else:
            print("Auto-offset requested but no tuple-matched TCP/UDP deltas found; using --alert-offset-seconds.")

    # Index alerts by 5-tuple (both directions) for faster matching
    alert_index = defaultdict(list)
    alert_dt_cache = [None] * len(alerts)
    for i, a in enumerate(alerts):
        ts = parse_alert_ts(a.get("timestamp", ""), year=2017)
        if ts is not None and offset_seconds:
            ts = ts - timedelta(seconds=offset_seconds)
        alert_dt_cache[i] = ts
        proto = norm_proto(a.get("proto"))
        k1 = (proto, a.get("src_ip"), a.get("src_port"), a.get("dst_ip"), a.get("dst_port"))
        k2 = (proto, a.get("dst_ip"), a.get("dst_port"), a.get("src_ip"), a.get("src_port"))
        keys = [k1, k2, (None, k1[1], k1[2], k1[3], k1[4]), (None, k2[1], k2[2], k2[3], k2[4])]
        for k in keys:
            alert_index[k].append(i)

    delta = timedelta(seconds=args.delta_seconds)
    alerted = np.zeros(len(merged), dtype=bool)
    matched_alerts = set()
    for idx, row in enumerate(merged.itertuples(index=False)):
        proto = getattr(row, "proto_num", None)
        if pd.isna(proto):
            proto = None
        else:
            proto = int(proto)
        key = (proto, getattr(row, "src_ip", None), getattr(row, "src_port", None), getattr(row, "dst_ip", None), getattr(row, "dst_port", None))
        cand = alert_index.get(key, []) + alert_index.get((None, key[1], key[2], key[3], key[4]), [])
        if not cand:
            continue
        start = getattr(row, "start_dt", None)
        end = getattr(row, "end_dt", None)
        if pd.isna(start):
            continue
        if pd.isna(end) or end is None:
            end = start
        for ai in cand:
            adt = alert_dt_cache[ai]
            if adt is None:
                continue
            if (start - delta) <= adt <= (end + delta):
                alerted[idx] = True
                matched_alerts.add(ai)
                break

    y_true = merged["target"].astype(int).to_numpy()
    y_pred = alerted.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "dataset": "CICIDS2017 Friday test subset",
        "rows_evaluated": int(len(merged)),
        "alerts_loaded": int(len(alerts)),
        "alerts_matched_to_flows": int(len(matched_alerts)),
        "flows_flagged": int(alerted.sum()),
        "time_tolerance_seconds": args.delta_seconds,
        "alert_offset_seconds_applied": offset_seconds,
        "alert_offset_seconds_auto_estimated": offset_est,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "notes": [
            "Snort alerts correlated to CICFlowMeter flows by 5-tuple (both directions) and timestamp window overlap.",
            "Alert timestamps parsed as 2017; an offset can be applied/auto-estimated to align Snort/pcap time to CICFlowMeter flow timestamps.",
            "Only Friday rows present in the held-out test split are evaluated for fair comparison with ML models.",
            "Decoder/L2 alerts (e.g., {eth}/{ARP}) lack usable 5-tuples and are not expected to correlate to flow records.",
        ],
    }
    out = ROOT / args.output
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
