#!/usr/bin/env python3
"""base (A1/A2) と integrated (B1) の中央値・差% を出す。1 本目を捨てる。
使い方: compare.py <arms_dir> <A1> <B1> <A2>
"""
import glob, json, os, statistics, sys

root = sys.argv[1]
a1, b1, a2 = sys.argv[2], sys.argv[3], sys.argv[4]


def load(label):
    out = {}
    for d in glob.glob(os.path.join(root, label + "*")):
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            recs = []
            for line in open(f, encoding="utf-8", errors="ignore"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            for r in recs:
                out.setdefault((r.get("kind"), r.get("prompt")), []).append(r)
    return out


A1, B1, A2 = load(a1), load(b1), load(a2)
keys = sorted(set(A1) | set(B1) | set(A2), key=lambda k: (k[1] or 0, k[0] or ""))
print("%-6s %-8s %10s %10s %10s %10s %10s %8s" % ("kind", "prompt", "A1", "A2", "A med", "B1", "B med", "B/A%"))
for k in keys:
    def med(d, field):
        v = [r[field] for r in d.get(k, []) if r.get(field)]
        return statistics.median(v[1:]) if len(v) > 1 else (v[0] if v else 0)
    a1p, a2p, b1p = med(A1, "prefill_tps"), med(A2, "prefill_tps"), med(B1, "prefill_tps")
    a1d, a2d, b1d = med(A1, "decode_tps"), med(A2, "decode_tps"), med(B1, "decode_tps")
    amed_p = statistics.median([x for x in (a1p, a2p) if x])
    amed_d = statistics.median([x for x in (a1d, a2d) if x])
    print("%-6s %-8s %10.1f %10.1f %10.1f %10.1f %10.1f %+7.1f%%  (prefill)" % (
        k[0], k[1], a1p, a2p, amed_p, b1p, b1p, (b1p / amed_p - 1) * 100 if amed_p else 0))
    if b1d or amed_d:
        print("%-6s %-8s %10.2f %10.2f %10.2f %10.2f %10.2f %+7.1f%%  (decode)" % (
            k[0], k[1], a1d, a2d, amed_d, b1d, b1d, (b1d / amed_d - 1) * 100 if amed_d else 0))
