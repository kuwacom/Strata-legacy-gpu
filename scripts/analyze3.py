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
labels = (["4K/code"] * 3 + ["4K/prose"] * 3 + ["24K/code"] * 3 + ["24K/prose"] * 3 +
          ["30K/code"] * 3 + ["30K/prose"] * 3)
groups = []
for l in labels:
    if l not in groups:
        groups.append(l)

names = [os.path.basename(a.rstrip("/")) for a in arms]
print("%-10s" % "size/kind" + "".join(" %24s" % n for n in names) + "   B/A1   B/A2   B/Amed")
for g in groups:
    idx = [i for i, l in enumerate(labels) if l == g]
    line = "%-10s" % g
    pre, dec = {}, {}
    for a in arms:
        rs = data[a]
        sel = [rs[i] for i in idx if i < len(rs)]
        pre[a] = med([r["prefill_tps"] for r in sel[1:]])
        dec[a] = med([r["decode_tps"] for r in sel[1:]])
        line += " %12.1f / %5.2f (%d)" % (pre[a], dec[a], len(sel))
    if len(arms) >= 3:
        amed = statistics.median([pre[arms[0]], pre[arms[2]]])
        line += "  %+6.2f%% %+6.2f%% %+6.2f%%" % (100 * (pre[arms[1]] / pre[arms[0]] - 1),
                                                   100 * (pre[arms[1]] / pre[arms[2]] - 1),
                                                   100 * (pre[arms[1]] / amed - 1))
    print(line)
print()
print("decode medians and B vs the A median:")
for g in groups:
    idx = [i for i, l in enumerate(labels) if l == g]
    vals = {}
    for a in arms:
        rs = data[a]
        sel = [rs[i] for i in idx if i < len(rs)]
        vals[a] = med([r["decode_tps"] for r in sel[1:]])
    line = "%-10s" % g + "".join(" %6.2f" % vals[a] for a in arms)
    if len(arms) >= 3:
        amed = statistics.median([vals[arms[0]], vals[arms[2]]])
        line += "   B/Amed %+6.2f%%" % (100 * (vals[arms[1]] / amed - 1))
    print(line)
print()
for a in arms:
    print(" ", os.path.basename(a.rstrip("/")))
    for i, r in enumerate(data[a]):
        print("    %2d %-10s %6d %7.1f %6.2f" % (i, labels[i] if i < len(labels) else "?", r["prompt"], r["prefill_tps"], r["decode_tps"]))
