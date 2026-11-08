#!/usr/bin/env python3
# bench jsonl の A-B-A 集計: (size, kind) ごとに rep 0 を捨てて中央値、B の A 比 % を出す
# 使い方: summarize_prefill.py base=path base2=path mmq=path
import json, statistics, sys

def load(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows

def size_of(p):
    return 24000 if p < 60000 else 128000

arms = []
for a in sys.argv[1:]:
    name, path = a.split("=", 1)
    arms.append((name, load(path)))

# セル: (size, kind) -> 出現順
cell = {}
for name, rows in arms:
    for r in rows:
        k = (size_of(r.get("prompt", 0)), r.get("kind"))
        cell.setdefault(k, {}).setdefault(name, []).append(r)

print(f"{'size':>7} {'kind':<6} " + " ".join(f"{n:>22}" for n, _ in arms))
for k in sorted(cell):
    line = f"{k[0]:>7} {k[1]:<6} "
    for name, _ in arms:
        rs = cell[k].get(name, [])
        vals = rs[1:] if len(rs) > 1 else rs   # 1 本目除外
        pre = [r["prefill_tps"] for r in vals if r.get("prefill_tps")]
        ttft = [r["ttft_s"] for r in vals if r.get("ttft_s")]
        if pre:
            line += f"{statistics.median(pre):8.1f}tps/{statistics.median(ttft):6.2f}s "
        else:
            line += f"{'--':>22} "
    print(line)

# B/A と B/A2 の比（中央値の中央値）
def med(name, k):
    rs = cell.get(k, {}).get(name, [])
    vals = rs[1:] if len(rs) > 1 else rs
    pre = [r["prefill_tps"] for r in vals if r.get("prefill_tps")]
    return statistics.median(pre) if pre else None

print()
for k in sorted(cell):
    b = med("mmq", k)
    a = med("base", k)
    a2 = med("base2", k)
    if b and a:
        ref = statistics.median([x for x in (a, a2) if x])
        print(f"{k[0]:>7} {k[1]:<6} base {a:8.1f} base2 {a2 if a2 else 0:8.1f} mmq {b:8.1f}  mmq vs ref {100*(b/ref-1):+.1f}%")
