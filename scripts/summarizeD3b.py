"""v3b draft-conf 集計: bench-<arm>.jsonl を固定順（4K/24K x code/prose x reps）で切り出す。

bench-decode.py は 1 行 = 1 リクエストで、sizes, kinds, reps の固定順で出すので
行の並びで条件を切り出す（summarize-v3.py と同じ考え方）。
使い方: summarizeD3b.py <results dir> <armA> [<armB> ...] [--reps 3]
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
            kept = rs[1:] if len(rs) > 1 else rs

            def med(f):
                return statistics.median([r[f] for r in kept]) if kept else None
            cond.setdefault(f"{size}/{kind}", {})[arm] = {
                "med": med("decode_tps"),
                "ttft": med("ttft_s"),
                "all": [r["decode_tps"] for r in rs],
            }

hdr = f"{'cond':10s}" + "".join(f"{a:>26s}" for a in arms)
print(hdr)
base = arms[0]
for c in [f"{s}/{k}" for s in sizes for k in kinds]:
    line = f"{c:10s}"
    vals = {}
    for a in arms:
        d = cond.get(c, {}).get(a)
        if not d or d["med"] is None:
            line += f"{'-':>26s}"
            continue
        vals[a] = d["med"]
        line += f"{d['med']:8.2f} tps ttft{d['ttft']:6.1f}s {'':>4s}"
    b = vals.get(base)
    for a in arms[1:]:
        v = vals.get(a)
        if v and b:
            line += f"  {a} vs {base}: {(v / b - 1) * 100:+.1f}%"
    print(line)

print("\n生の run（1 本目が先頭）:")
for arm in arms:
    for c in [f"{s}/{k}" for s in sizes for k in kinds]:
        rs = cond.get(c, {}).get(arm, {}).get("all", [])
        if rs:
            print(f"  {arm} {c}: " + " ".join(f"{x:.2f}" for x in rs))
