import glob, json, os, statistics, sys

def load(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "bench-*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def med(v):
    return statistics.median(v) if v else float("nan")

arms = sys.argv[1:]
data = {a: load(a) for a in arms}
# bench.py's order: sizes 4000,24000,30000 x kind code x repeat 3 -> 9 rows
labels = ["4K"] * 3 + ["24K"] * 3 + ["30K"] * 3
print("%-4s" % "size", end="")
for a in arms:
    print(" %26s" % os.path.basename(a.rstrip("/")), end="")
print("   B-A% (B vs A1/A2)")
for g, name in enumerate(["4K", "24K", "30K"]):
    idx = [i for i in range(len(labels)) if labels[i] == name]
    line = "%-4s" % name
    pre = {}
    dec = {}
    for a in arms:
        rs = data[a]
        sel = [rs[i] for i in idx if i < len(rs)]
        pre[a] = med([r["prefill_tps"] for r in sel[1:]])
        dec[a] = med([r["decode_tps"] for r in sel[1:]])
        line += " %10.1f / %5.2f (%d)" % (pre[a], dec[a], len(sel))
    if len(arms) >= 2:
        line += "   %+6.2f%%" % (100.0 * (pre[arms[1]] / pre[arms[0]] - 1.0))
    print(line)
print()
print("decode medians:")
for g, name in enumerate(["4K", "24K", "30K"]):
    idx = [i for i in range(len(labels)) if labels[i] == name]
    line = "%-4s" % name
    for a in arms:
        rs = data[a]
        sel = [rs[i] for i in idx if i < len(rs)]
        line += " %6.2f" % med([r["decode_tps"] for r in sel[1:]])
    if len(arms) >= 2:
        line += "   B-A %+6.2f%%" % (100.0 * (med([data[arms[1]][i]["decode_tps"] for i in idx if i < len(data[arms[1]])][1:]) /
                                            med([data[arms[0]][i]["decode_tps"] for i in idx if i < len(data[arms[0]])][1:]) - 1.0))
    print(line)
print()
print("per-rep detail (prompt tokens, prefill, decode):")
for a in arms:
    print(" ", os.path.basename(a.rstrip("/")))
    for i, r in enumerate(data[a]):
        print("    %d %-6s %6d %7.1f %6.2f" % (i, labels[i] if i < len(labels) else "?", r["prompt"], r["prefill_tps"], r["decode_tps"]))
