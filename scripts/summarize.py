#!/usr/bin/env python3
"""arm-integrated.sh の jsonl から 1 本目を捨てた中央値と差% を出す。
使い方: summarize.py <dir>  (arms/<label>-<mode>/ を走査)
"""
import glob, json, os, statistics, sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/work/results/integrated/arms")
rows = {}
for d in sorted(glob.glob(os.path.join(root, "*"))):
    label = os.path.basename(d)
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        recs = []
        for line in open(f, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        for r in recs:
            key = (label.split("-")[0], r.get("kind"), r.get("prompt"))
            rows.setdefault(key, []).append((os.path.basename(f), r))

for key in sorted(rows):
    vals = rows[key]
    pref = [v["prefill_tps"] for _, v in vals]
    dec = [v["decode_tps"] for _, v in vals if v.get("decode_tps")]
    ttft = [v["ttft_s"] for _, v in vals]
    print(key, "n=%d" % len(vals))
    print("  prefill", pref, "median(all)=%.1f median(no1st)=%.1f" % (
        statistics.median(pref), statistics.median(pref[1:]) if len(pref) > 1 else pref[0]))
    if dec:
        print("  decode ", dec, "median(all)=%.2f median(no1st)=%.2f" % (
            statistics.median(dec), statistics.median(dec[1:]) if len(dec) > 1 else dec[0]))
    print("  ttft   ", ttft)
