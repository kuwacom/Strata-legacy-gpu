"""A/B の集計: bench-<arm>.jsonl を読み、kind ごとに 1 本目を除いた中央値と差% を出す。

使い方: analyze.py <results dir> <arm> [<arm> ...]
"""
import json
import os
import statistics
import sys

res = sys.argv[1]
arms = sys.argv[2:]

data = {}
for arm in arms:
    p = os.path.join(res, f"bench-{arm}.jsonl")
    rows = []
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    data[arm] = rows


def key(r):
    return (r["prompt"] // 1000, r["kind"])


def med(rows, field):
    return statistics.median([r[field] for r in rows]) if rows else None


groups = {}
for arm, rows in data.items():
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    for k, rs in by.items():
        rs = sorted(rs, key=lambda r: r.get("rep", 0))
        kept = rs[1:] if len(rs) > 1 else rs   # 1 本目除外
        groups.setdefault(k, {})[arm] = (rs, kept)

print(f"{'condition':34s} " + " ".join(f"{a:>22s}" for a in arms))
for k in sorted(groups):
    line = f"{str(k):34s} "
    for arm in arms:
        rs, kept = groups[k].get(arm, ([], []))
        if not rs:
            line += f"{'-':>22s} "
            continue
        allmed = med(rs, "decode_tps")
        keptmed = med(kept, "decode_tps")
        ttft = med(kept, "ttft_s")
        line += f"{keptmed:7.2f}(n={len(kept)},all {allmed:.1f},ttft {ttft:.1f}) "
    print(line)

# 差%（base を基準に、1 本目除外の中央値）
if "base" in arms:
    print("\ndiff vs base (decode_tps, first rep excluded):")
    for k in sorted(groups):
        base = groups[k].get("base")
        if not base or not base[1]:
            continue
        b = med(base[1], "decode_tps")
        parts = []
        for arm in arms:
            if arm == "base":
                continue
            rs, kept = groups[k].get(arm, ([], []))
            if not kept:
                continue
            m = med(kept, "decode_tps")
            parts.append(f"{arm} {100.0 * (m - b) / b:+.1f}%")
        print(f"{str(k):34s} base {b:6.2f}  " + "  ".join(parts))

