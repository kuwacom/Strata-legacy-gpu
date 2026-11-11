#!/usr/bin/env python3
"""2 つのエンジンログの prefill 位相内訳を比べる（GPU timeline と位相ごとの ms / 差 / 差%）"""
import re
import sys

PAT = re.compile(r"prefill timing: (\d+) tokens, GPU timeline (\d+) ms.*?: (.*)$")
PART = re.compile(r"([^()]+?) (\d+) \(([\d.]+)%\)")


def parse(path, want_tokens=None):
    best = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        tok = int(m.group(1))
        if want_tokens is not None and tok != want_tokens:
            continue
        phase = {}
        for mm in PART.finditer(m.group(3)):
            phase[mm.group(1).strip()] = int(mm.group(2))
        best = (tok, int(m.group(2)), phase)
    return best


def main():
    a = parse(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    b = parse(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else None)
    if not a or not b:
        print("no matching prefill timing line")
        return 1
    ta, ga, pa = a
    tb, gb, pb = b
    print(f"A: {sys.argv[1]} tokens={ta} timeline={ga} ms")
    print(f"B: {sys.argv[2]} tokens={tb} timeline={gb} ms")
    if ta != tb:
        print("token counts differ: not comparable")
    print(f"timeline {gb - ga:+d} ms ({(gb - ga) / ga * 100:+.2f}%)")
    order = [k for k in pa] + [k for k in pb if k not in pa]
    print(f"{'phase':28s} {'A ms':>8s} {'B ms':>8s} {'diff':>8s} {'pct':>8s}")
    for k in order:
        va = pa.get(k, 0)
        vb = pb.get(k, 0)
        if va == 0 and vb == 0:
            continue
        pct = f"{(vb - va) / va * 100:+.2f}%" if va else "new"
        print(f"{k:28s} {va:8d} {vb:8d} {vb - va:+8d} {pct:>8s}")


if __name__ == "__main__":
    sys.exit(main())
