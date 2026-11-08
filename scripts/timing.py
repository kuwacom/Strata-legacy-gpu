"""timing-<arm>.txt の "strata decode timing:" 行を数値にして、条件ごとの中央値を出す。

フィルタ: windows >= 50（bench の 256 token decode）。53 token の起動プローブ（1 window）は捨てる。
使い方: timing.py <results dir> <arm> [...]
"""
import os
import re
import statistics
import sys

res = sys.argv[1]
arms = sys.argv[2:]
pat = re.compile(
    r"(\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = adapt wait ([\d.]+) \+ verify ([\d.]+) "
    r"\(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan ([\d.]+) actq ([\d.]+) jobs ([\d.]+) CPU ([\d.]+)\] \+ stage ([\d.]+)\) "
    r"\+ commit/emit ([\d.]+) \+ draft ([\d.]+) \+ adapt join ([\d.]+) \+ other ([\d.]+); per layer-window: CPU experts ([\d.]+) "
    r"\(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+); ([\d.]+) swaps/window")
keys = ["win", "T", "tokwin", "ms", "adapt", "verify", "wait", "host", "plan", "actq", "jobs", "cpu",
        "stage", "commit", "draft", "join", "other", "cpu_exp", "entries", "vram", "pcie", "swaps"]


def load(arm):
    p = os.path.join(res, f"timing-{arm}.txt")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, errors="ignore"):
        m = pat.search(line)
        if m:
            d = dict(zip(keys, [float(x) for x in m.groups()]))
            if d["win"] >= 50:
                out.append(d)
    return out


rows = {a: load(a) for a in arms}
cols = ["win", "tokwin", "ms", "wait", "host", "cpu", "draft", "cpu_exp", "vram", "pcie", "swaps"]
print(f"{'arm':8s}{'n':>4s}" + "".join(f"{c:>9s}" for c in cols))
for a in arms:
    rs = rows[a]
    if not rs:
        print(f"{a:8s}{0:>4d}  (no timing lines)")
        continue
    print(f"{a:8s}{len(rs):>4d}" + "".join(f"{statistics.median([r[c] for r in rs]):9.2f}" for c in cols))
print("\n(wait = GPU-reach wait, host = per-layer host, cpu = CPU experts; すべて ms/window の内訳)")
for a in arms:
    rs = rows[a]
    if rs:
        share = statistics.median([r["wait"] / r["ms"] * 100 for r in rs])
        print(f"  {a}: GPU-reach wait share {share:.1f}% of ms/window")
