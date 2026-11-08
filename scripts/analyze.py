#!/usr/bin/env python3
"""統合 A-B-A の集計。bench.py は rep ごとにプロンプト長が変わるので、jsonl の並び順で
(size, kind) を復元する。decode は 4K code x3 / 4K prose x3 / 24K code x3 / 24K prose x3。
1 本目を捨てて残りの中央値を出す。
使い方: analyze.py <arms_dir>
"""
import glob, json, os, statistics, sys

root = sys.argv[1]
labels = ["A1", "A2", "B1"]

GROUPS = [("4K", "code"), ("4K", "prose"), ("24K", "code"), ("24K", "prose")]


def read_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def arm_data(label):
    d = {}
    dec_files = sorted(glob.glob(os.path.join(root, label + "-speed", "bench-*.jsonl")))
    recs = []
    for f in dec_files:
        recs += read_jsonl(f)
    d["decode"] = {}
    for i, (size, kind) in enumerate(GROUPS):
        grp = recs[i * 3:(i + 1) * 3]
        d["decode"][(size, kind)] = grp
    for key, pat in (("prefill24", "prefill-24k-*.jsonl"), ("prefill128", "prefill-128k-*.jsonl")):
        recs = []
        for f in sorted(glob.glob(os.path.join(root, label + "-speed", pat))):
            recs += read_jsonl(f)
        d[key] = recs
    return d


def med(recs, field):
    vals = [r[field] for r in recs if r.get(field)]
    vals = vals[1:] if len(vals) > 1 else vals  # 1 本目を捨てる
    return statistics.median(vals) if vals else 0.0


data = {l: arm_data(l) for l in labels if glob.glob(os.path.join(root, l + "-speed"))}
print("== decode (tok/s, median of reps after the first) ==")
print("%-4s %-6s %8s %8s %8s %8s %8s %8s %7s" % ("size", "kind", "A1", "A2", "Amed", "B1", "Bmed", "B/A%", "noise%"))
for size, kind in GROUPS:
    a1 = med(data.get("A1", {}).get("decode", {}).get((size, kind), []), "decode_tps")
    a2 = med(data.get("A2", {}).get("decode", {}).get((size, kind), []), "decode_tps")
    b1 = med(data.get("B1", {}).get("decode", {}).get((size, kind), []), "decode_tps")
    amed = statistics.median([x for x in (a1, a2) if x])
    noise = abs(a2 - a1) / a1 * 100 if a1 else 0
    print("%-4s %-6s %8.2f %8.2f %8.2f %8.2f %8.2f %+7.1f %+7.1f" % (
        size, kind, a1, a2, amed, b1, b1, (b1 / amed - 1) * 100 if amed else 0, noise))
print("== prefill (tok/s, median after the first) ==")
for key, name in (("prefill24", "24K"), ("prefill128", "128K")):
    a1 = med(data.get("A1", {}).get(key, []), "prefill_tps")
    a2 = med(data.get("A2", {}).get(key, []), "prefill_tps")
    b1 = med(data.get("B1", {}).get(key, []), "prefill_tps")
    amed = statistics.median([x for x in (a1, a2) if x])
    noise = abs(a2 - a1) / a1 * 100 if a1 else 0
    print("%-5s %8.1f %8.1f %8.1f %8.1f %8.1f %+7.1f %+7.1f" % (
        name, a1, a2, amed, b1, b1, (b1 / amed - 1) * 100 if amed else 0, noise))
print("== TTFT (s) ==")
for key, name in (("prefill24", "24K"), ("prefill128", "128K")):
    for l in labels:
        recs = data.get(l, {}).get(key, [])
        print(name, l, [round(r["ttft_s"], 1) for r in recs], "prompt", [r["prompt"] for r in recs])
