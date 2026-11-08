#!/usr/bin/env python3
# 指定 pos の共通レコードだけ KL を出す（verify window と bench の記録を避ける）
# 使い方: kl_at.py base.bin cand.bin pos1,pos2,...
import math, struct, sys

want = set(int(x) for x in sys.argv[3].split(","))

def read_records(path):
    out = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            magic, T, nv, rec = struct.unpack("<4i", hdr)
            if magic != 0x53474F4C:
                break
            pos, = struct.unpack("<q", f.read(8))
            buf = f.read(4 * T * nv)
            if pos in want:
                row = struct.unpack("<%df" % (T * nv), buf)
                out[(pos, T)] = (rec, row[:nv])
    return out

def log_softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    ls = math.log(math.fsum(e))
    return [v - m - ls for v in x]

a = read_records(sys.argv[1])
b = read_records(sys.argv[2])
for k in sorted(set(a) & set(b)):
    la = log_softmax(a[k][1])
    lb = log_softmax(b[k][1])
    kl = math.fsum(math.exp(x) * (x - y) for x, y in zip(la, lb))
    top_a = max(range(len(a[k][1])), key=lambda i: a[k][1][i])
    top_b = max(range(len(b[k][1])), key=lambda i: b[k][1][i])
    print("pos=%d T=%d rec=%d KL=%.6g bitwise=%s top1 base=%d cand=%d same=%s"
          % (k[0], k[1], a[k][0], kl, a[k][1] == b[k][1], top_a, top_b, top_a == top_b))
