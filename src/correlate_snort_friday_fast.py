#!/usr/bin/env python3
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

ALERT_TIME_FORMATS = [
    "%m/%d/%y-%H:%M:%S.%f",
    "%m/%d/%y-%H:%M:%S",
    "%m/%d-%H:%M:%S.%f",
    "%m/%d-%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
]


def norm_proto_to_num(p):
    if p is None:
        return None
    p = str(p).upper()
    if p == "TCP" or p == "6":
        return 6
    if p == "UDP" or p == "17":
        return 17
    if p == "ICMP" or p == "1":
        return 1
    return None


def parse_alert_ts(s, year=2017):
    if not s:
        return None
    s = s.strip()
    for fmt in ALERT_TIME_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 100:
                dt = dt.replace(year=year)
            elif dt.year < year and "%y" not in fmt:
                dt = dt.replace(year=year)
            return dt
        except Exception:
            continue
    return None


def norm_proto(p):
    if p is None:
        return None
    p = str(p).upper()
    if p == "6":
        return "TCP"
    if p == "17":
        return "UDP"
    if p == "1":
        return "ICMP"
    if p in ("TCP", "UDP", "ICMP", "IP", "ETH", "ARP"):
        return p
    return p


def load_friday_test():
    df = pickle.load(open(DATA_PROCESSED / "cleaned_flows.pkl", "rb"))
    split = np.load(DATA_PROCESSED / "split_indices.npz")
    test = df.iloc[split["test"]].copy()
    fri = test[test["source_file"].str.contains("Friday", case=False, na=False)].copy()
    frames = []
    for name in FRIDAY_FILES:
        path = LABELLED_DIR / name
        if not path.exists():
            continue
        m = pd.read_csv(path, low_memory=False)
        m.columns = [c.strip() for c in m.columns]
        m = m.rename(
            columns={
                "Source IP": "src_ip",
                "Source Port": "src_port",
                "Destination IP": "dst_ip",
                "Destination Port": "dst_port",
                "Protocol": "proto",
                "Timestamp": "timestamp",
                "Flow Duration": "flow_duration",
            }
        )
        m["source_file"] = name
        m["source_row"] = np.arange(1, len(m) + 1)
        frames.append(m)
    if not frames:
        raise SystemExit("No Friday labelled files found")
    meta = pd.concat(frames, ignore_index=True)
    for col in ["src_port", "dst_port", "flow_duration"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta["proto_num"] = meta["proto"].apply(norm_proto_to_num)
    meta["start_dt"] = pd.to_datetime(meta["timestamp"], errors="coerce", dayfirst=True)
    meta["end_dt"] = meta["start_dt"] + pd.to_timedelta(meta["flow_duration"].fillna(0), unit="us")
    needed = meta[["source_file", "source_row", "src_ip", "src_port", "dst_ip", "dst_port", "proto_num", "start_dt", "end_dt"]].copy()
    merged = fri.merge(needed, on=["source_file", "source_row"], how="left")
    return merged


def main():
    ap = argparse.ArgumentParser(description="Fast correlation of Snort alerts to Friday test flows")
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--delta-seconds", type=float, default=60.0)
    ap.add_argument("--alert-offset-seconds", type=float, default=0.0)
    ap.add_argument("--auto-offset", action="store_true")
    ap.add_argument("--max-auto-check", type=int, default=10000)
    ap.add_argument("-o", "--output", default="reports/snort_friday_evaluation_fast.json")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    merged = load_friday_test()
    print(f"Friday test flows: {len(merged)}, missing metadata: {merged['start_dt'].isna().sum()}")

    payload = json.loads(Path(args.alerts).read_text())
    alerts = payload.get("alerts", [])
    print(f"Loaded {len(alerts)} parsed alerts")

    offset_seconds = float(args.alert_offset_seconds)
    offset_est = None

    # Build flow index by tuple (with proto-specific and proto-agnostic keys)
    flow_index = defaultdict(list)
    for i, row in enumerate(merged.itertuples(index=False)):
        try:
            proto = int(getattr(row, "proto_num")) if pd.notna(getattr(row, "proto_num")) else None
        except Exception:
            proto = None
        sip = getattr(row, "src_ip")
        dip = getattr(row, "dst_ip")
        sp = int(getattr(row, "src_port")) if pd.notna(getattr(row, "src_port")) else None
        dp = int(getattr(row, "dst_port")) if pd.notna(getattr(row, "dst_port")) else None
        start = getattr(row, "start_dt")
        end = getattr(row, "end_dt")
        if pd.isna(start):
            continue
        if pd.isna(end):
            end = start
        entry = (i, start, end)
        for pn in [proto, None]:
            flow_index[(pn, sip, sp, dip, dp)].append(entry)
            flow_index[(pn, dip, dp, sip, sp)].append(entry)

    if args.auto_offset:
        deltas = []
        checked = 0
        for a in alerts:
            if a.get("proto") not in ("TCP", "UDP") or not a.get("src_port"):
                continue
            pn = norm_proto_to_num(a.get("proto"))
            try:
                sp = int(a.get("src_port")); dp = int(a.get("dst_port"))
            except Exception:
                continue
            cand_keys = [
                (pn, a.get("src_ip"), sp, a.get("dst_ip"), dp),
                (pn, a.get("dst_ip"), dp, a.get("src_ip"), sp),
                (None, a.get("src_ip"), sp, a.get("dst_ip"), dp),
                (None, a.get("dst_ip"), dp, a.get("src_ip"), sp),
            ]
            found_flow = False
            for kk in cand_keys:
                if kk in flow_index and flow_index[kk]:
                    adt = parse_alert_ts(a.get("timestamp", ""), year=2017)
                    if adt is None:
                        break
                    fstart = flow_index[kk][0][1]
                    deltas.append((adt - fstart).total_seconds())
                    found_flow = True
                    break
            if found_flow:
                checked += 1
                if checked >= args.max_auto_check:
                    break
        if deltas:
            offset_est = float(pd.Series(deltas).median())
            offset_seconds = offset_est
            print(f"Auto-offset median: {offset_est:.3f}s (checked {checked}, deltas {len(deltas)})")
        else:
            print("Auto-offset: no matches found")

    delta = timedelta(seconds=args.delta_seconds)
    alerted_idx = set()
    matched_alerts = 0
    alerts_with_dt = 0
    for a in alerts:
        adt = parse_alert_ts(a.get("timestamp", ""), year=2017)
        if adt is None:
            continue
        alerts_with_dt += 1
        if offset_seconds:
            adt = adt - timedelta(seconds=offset_seconds)
        proto = norm_proto(a.get("proto"))
        pnum = norm_proto_to_num(proto)
        sip = a.get("src_ip"); dip = a.get("dst_ip")
        sp = a.get("src_port"); dp = a.get("dst_port")
        if sp is not None:
            try:
                sp = int(sp)
            except Exception:
                sp = None
        if dp is not None:
            try:
                dp = int(dp)
            except Exception:
                dp = None
        keys = []
        for pn in [pnum, None]:
            keys.append((pn, sip, sp, dip, dp))
            keys.append((pn, dip, dp, sip, sp))
        hit = False
        for k in keys:
            if k in flow_index:
                for (fi, fstart, fend) in flow_index[k]:
                    if (fstart - delta) <= adt <= (fend + delta):
                        alerted_idx.add(fi)
                        hit = True
                        break
                if hit:
                    break
        if hit:
            matched_alerts += 1

    y_true = merged["target"].astype(int).to_numpy()
    y_pred = np.zeros(len(merged), dtype=int)
    for fi in alerted_idx:
        y_pred[fi] = 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "dataset": "CICIDS2017 Friday test subset",
        "rows_evaluated": int(len(merged)),
        "alerts_loaded": int(len(alerts)),
        "alerts_matched_to_flows": int(matched_alerts),
        "flows_flagged": int(y_pred.sum()),
        "time_tolerance_seconds": args.delta_seconds,
        "alert_offset_seconds_applied": offset_seconds,
        "alert_offset_seconds_auto_estimated": offset_est,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "notes": [
            "Fast correlation by flow-indexed tuples with optional timestamp offset.",
            "Decoder/L2 alerts without flow tuples may not match.",
        ],
    }
    out = Path(args.output)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
