#!/usr/bin/env python3
"""Parse Snort 3 alert_fast output into structured records."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Snort 3 alert_fast variants:
# 07/07-10:15:00.123456 [**] [1:1000001:1] "msg" [**] [Classification: ...] [Priority: 1] {TCP} 1.2.3.4:12345 -> 5.6.7.8:80
# Decoder/L2 alerts may omit IP endpoints: {eth}  ->  or {ARP}  ->
ALERT_RE = re.compile(
    r"^(?P<ts>\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"\[\*\*\]\s+\[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\]\s+"
    r"\"(?P<msg>[^\"]*)\"\s+\[\*\*\]\s+"
    r"(?:\[Classification:[^\]]*\]\s*)?"
    r"(?:\[Priority:\s*(?P<prio>\d+)\]\s*)?"
    r"\{(?P<proto>[^}]+)\}"
    r"(?:\s+(?:(?P<src>[^\s:]+)(?::(?P<sport>\d+))?)?\s*->\s*(?:(?P<dst>[^\s:]+)(?::(?P<dport>\d+))?)?)?"
    r"\s*$"
)


def parse_file(path: Path) -> list[dict]:
    alerts = []
    bad = 0
    with path.open("r", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = ALERT_RE.search(line)
            if not m:
                bad += 1
                if bad <= 5:
                    alerts.append({"_unparsed": True, "line": i, "raw": line})
                continue
            d = m.groupdict()
            alerts.append(
                {
                    "timestamp": d["ts"],
                    "gid": int(d["gid"]),
                    "sid": int(d["sid"]),
                    "rev": int(d["rev"]),
                    "msg": d["msg"],
                    "priority": int(d["prio"]) if d.get("prio") else None,
                    "proto": d["proto"].strip(),
                    "src_ip": d["src"],
                    "src_port": int(d["sport"]) if d.get("sport") else None,
                    "dst_ip": d["dst"],
                    "dst_port": int(d["dport"]) if d.get("dport") else None,
                    "raw": line,
                }
            )
    return alerts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("alert_file", nargs="?", default="snort_out/alert_fast.txt")
    ap.add_argument("-o", "--output", default="reports/snort_alerts_parsed.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    in_path = (root / args.alert_file).resolve() if not Path(args.alert_file).is_absolute() else Path(args.alert_file)
    alerts = parse_file(in_path)
    out_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = [a for a in alerts if not a.get("_unparsed")]
    summary = {
        "source": str(in_path),
        "total_lines_parsed": len(parsed),
        "unparsed_or_skipped": sum(1 for a in alerts if a.get("_unparsed")),
        "sample": parsed[:5],
    }
    out_path.write_text(json.dumps({"summary": summary, "alerts": parsed}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {len(parsed)} alerts to {out_path}")


if __name__ == "__main__":
    main()
