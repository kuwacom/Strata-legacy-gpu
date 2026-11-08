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
keys = sorted({(r["prompt"], r["kind"]) for rs in data.values() for r in rs})
print("prompt tokens x kind:")
for a in arms:
    print("  %-4s %s" % (a, sorted({r["prompt"] for r in data[a]})))
print()
hdr = "%-22s" % "size/kind"
for a in arms:
    hdr += " %18s" % a
print(hdr + "   B-A%")
for k in keys:
    line = "%-22s" % ("%d/%s" % k)
    vals = {}
    for a in arms:
        rs = [r for r in data[a] if (r["prompt"], r["kind"]) == k]
        # first run excluded, median of the rest
        pre = [r["prefill_tps"] for r in rs[1:]]
        dec = [r["decode_tps"] for r in rs[1:]]
        vals[a] = (med(pre), med(dec), len(rs))
        line += " %8.1f/%5.2f(%d)" % (vals[a][0], vals[a][1], vals[a][2])
    if len(arms) >= 2 and vals[arms[0]][0] == vals[arms[0]][0] and vals[arms[1]][0] == vals[arms[1]][0]:
        line += "   %+6.2f%%" % (100.0 * (vals[arms[1]][0] / vals[arms[0]][0] - 1.0))
    print(line)
print()
print("decode diffs (B vs A):")
for k in keys:
    rs0 = [r for r in data[arms[0]] if (r["prompt"], r["kind"]) == k][1:]
    rs1 = [r for r in data[arms[1]] if (r["prompt"], r["kind"]) == k][1:]
    d0, d1 = med([r["decode_tps"] for r in rs0]), med([r["decode_tps"] for r in rs1])
    print("  %-18s %6.2f -> %6.2f  %+6.2f%%" % ("%d/%s" % k, d0, d1, 100.0 * (d1 / d0 - 1.0)))
