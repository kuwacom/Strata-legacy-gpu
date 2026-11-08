"""v3 の集計: bench-<arm>.jsonl を読み、条件ごとに 1 本目を除いた中央値を出す。

bench-decode.py は (size, kind, rep) の固定順で 1 行ずつ出す（プロンプト長は rep ごとに少し違う）ので、
行の並びで条件を切り出す（prompt//1000 でまとめると rep ごとに別グループになる）。
使い方: summarize-v3.py <results dir> <armA> <armB> ... [--reps 3]
"""
import json
import os
import statistics
import sys

args = sys.argv[1:]
reps = 3
if "--reps" in args:
    i = args.index("--reps")
    reps = int(args[i + 1])
    args = args[:i] + args[i + 2:]
res, arms = args[0], args[1:]
sizes = ["4K", "24K"]
kinds = ["code", "prose"]

data = {}
for arm in arms:
    p = os.path.join(res, f"bench-{arm}.jsonl")
    data[arm] = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

cond = {}
for arm, rows in data.items():
    if len(rows) != len(sizes) * len(kinds) * reps:
        print(f"# {arm}: {len(rows)} rows (expected {len(sizes)*len(kinds)*reps})")
    for si, size in enumerate(sizes):
        for ki, kind in enumerate(kinds):
            off = (si * len(kinds) + ki) * reps
            rs = rows[off:off + reps]

            def med(f):
                return statistics.median([r[f] for r in rs]) if rs else None
            cand = [r for r in rs if r["kind"] == kind]
            cond.setdefault(f"{size}/{kind}", {})[arm] = {
                "all": rs,
                "kept": cand[1:] if len(cand) > 1 else cand,
                "med_all": med("decode_tps"),
            }

hdr = f"{'cond':10s}" + "".join(f"{a:>30s}" for a in arms)
print(hdr)
for c in [f"{s}/{k}" for s in sizes for k in kinds]:
    line = f"{c:10s}"
    for a in arms:
        d = cond.get(c, {}).get(a)
        if not d or not d["all"]:
            line += f"{'-':>30s}"
            continue
        k = d["kept"]
        line += f"{statistics.median([r['decode_tps'] for r in k]):7.2f} ttft{statistics.median([r['ttft_s'] for r in k]):6.1f} {'':>8s}"
    print(line)

print("\n消した 1 本目 (decode_tps, 1 本目 / 残り中央値):")
for c in [f"{s}/{k}" for s in sizes for k in kinds]:
    parts = []
    for a in arms:
        d = cond.get(c, {}).get(a)
        if not d or not d["all"]:
            continue
        first = d["all"][0]["decode_tps"]
        kept = statistics.median([r["decode_tps"] for r in d["kept"]])
        parts.append(f"{a} {first:.1f}/{kept:.1f}")
    print(f"  {c:10s} " + "  ".join(parts))

print(f"\ndecode_tps diff vs {arms[0]} (median of 2/3 本目):")
for c in [f"{s}/{k}" for s in sizes for k in kinds]:
    b = cond.get(c, {}).get(arms[0])
    if not b or not b["all"]:
        continue
    bm = statistics.median([r["decode_tps"] for r in b["kept"]])
    parts = []
    for a in arms[1:]:
        d = cond.get(c, {}).get(a)
        if not d or not d["all"]:
            continue
        m = statistics.median([r["decode_tps"] for r in d["kept"]])
        parts.append(f"{a} {100.0*(m-bm)/bm:+.1f}%")
    print(f"  {c:10s} {arms[0]} {bm:6.2f}  " + "  ".join(parts))

print("\n全条件の中央値 (median of the four condition medians):")
for a in arms:
    vals = [statistics.median([r["decode_tps"] for r in cond[c][a]["kept"]])
            for c in cond if a in cond[c] and cond[c][a]["all"]]
    if vals:
        print(f"  {a:8s} {statistics.median(vals):6.2f} tok/s  ({len(vals)} conditions)")
