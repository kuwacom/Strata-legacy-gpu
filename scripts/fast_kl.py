#!/usr/bin/env python3
# STRATA_LOGITS_DUMP の共通レコードで KL(last row) を出す（numpy 無し）
# 使い方: fast_kl.py base.bin cand.bin [pos_max]
import math, struct, sys

def read_records(path, pos_max):
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
            if pos <= pos_max:
                row = struct.unpack("<%df" % (T * nv), buf)
                out[(pos, T)] = (rec, row[:nv], row)
            # skip rows past pos_max without unpacking (seek is not possible on a pipe, but the read above consumed it)
    return out

def log_softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = math.fsum(e)
    ls = math.log(s)
    return [v - m - ls for v in x]

a = read_records(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 10**9)
b = read_records(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10**9)
common = sorted(set(a) & set(b))
print("records base=%d cand=%d common=%d" % (len(a), len(b), len(common)))
kls = []
for k in common:
    la = log_softmax(a[k][1])
    lb = log_softmax(b[k][1])
    kl = 0.0
    for i in range(len(la)):
        p = math.exp(la[i])
        if p > 0.0:
            kl += p * (la[i] - lb[i])
    eq = int(a[k][1] == b[k][1])
    kls.append((kl, k, eq))
kls.sort(reverse=True)
print("top KL:")
for kl, k, eq in kls[:8]:
    print("  pos=%d T=%d KL=%.6g bitwise=%s" % (k[0], k[1], kl, eq))
vals = sorted(x[0] for x in kls)
print("median KL: %.6g" % vals[len(vals) // 2])
print("bitwise rows: %d / %d" % (sum(1 for x in kls if x[2]), len(kls)))
print("KL > 0.01: %d / %d" % (sum(1 for x in kls if x[0] > 0.01), len(kls)))
