"""v3b PATTN 集計: bench-<label>.jsonl を (target_tokens, kind) で束ね、run 0 を捨てて中央値を出す。

使い方: summarizeV3c.py <results dir> <label> [<label> ...]
出力: 条件 x アーム の prefill_tps 中央値（A 基準の差% 付き）と TTFT 中央値
"""
import json
import os
import statistics
import sys

res = sys.argv[1]
arms = sys.argv[2:]
conds = [("24000", "prose"), ("24000", "code"), ("128000", "prose"), ("128000", "code")]

data = {}
for arm in arms:
    p = os.path.join(res, f"{arm}-bench.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
    g = {}
    for r in rows:
        g.setdefault((str(r["target_tokens"]), r.get("kind", "?")), []).append(r)
    data[arm] = g


def med(arm, cond, field):
    rs = data.get(arm, {}).get(cond, [])
    kept = rs[1:] if len(rs) > 2 else rs  # run 0 を捨てる（3 回以上のとき）
    return statistics.median([r[field] for r in kept]) if kept else None


hdr = f"{'cond':16s}" + "".join(f"{a:>26s}" for a in arms)
print(hdr)
base = arms[0]
for cond in conds:
    line = f"{cond[0]+'/'+cond[1]:16s}"
    vals = {}
    for arm in arms:
        m = med(arm, cond, "prefill_tps")
        t = med(arm, cond, "ttft_s")
        vals[arm] = m
        cell = "-" if m is None else f"{m:7.1f} tps ttft {t:6.1f}s"
        line += f"{cell:>26s}"
    if vals.get(base) and vals.get(arms[1] if len(arms) > 1 else base):
        b = vals[base]
        for arm in arms[1:]:
            v = vals[arm]
            if v and b:
                line += f"  {arm} vs {base}: {(v / b - 1) * 100:+.1f}%"
    print(line)

print("\n生の run（run 0 = 1 本目・除外対象）:")
for arm in arms:
    for cond in conds:
        rs = data.get(arm, {}).get(cond, [])
        if rs:
            s = " ".join(f"{r['prefill_tps']:.1f}" for r in rs)
            print(f"  {arm} {cond[0]}/{cond[1]}: {s}")
