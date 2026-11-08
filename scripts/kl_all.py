#!/usr/bin/env python3
# STRATA_LOGITS_DUMP の全レコードを (pos, T) で突き合わせる。row 0（窓の先頭）の bitwise 差と KL を出す
# 使い方: kl_all.py base.bin cand.bin
import math
import struct
import sys

def read_records(path):
    out = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            magic, T, nv, rec = struct.unpack("<4i", hdr)
            if magic != 0x53474F4C:
                print("bad magic", magic)
                break
            pos, = struct.unpack("<q", f.read(8))
            row = struct.unpack("<%df" % (T * nv), f.read(4 * T * nv))
            out[(pos, T)] = (rec, row[:nv], row)
    return out

def log_softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [math.log(v / s) for v in e]

a = read_records(sys.argv[1])
b = read_records(sys.argv[2])
common = sorted(set(a) & set(b))
only_a = sorted(set(a) - set(b))
only_b = sorted(set(b) - set(a))
print("records base=%d cand=%d common=%d only_base=%d only_cand=%d" % (len(a), len(b), len(common), len(only_a), len(only_b)))
if only_a[:10]:
    print("only base:", only_a[:10])
if only_b[:10]:
    print("only cand:", only_b[:10])

worst = 0.0
worst_pos = None
worst_kl = 0.0
worst_kl_pos = None
agree = 0
for k in common:
    _, r0a, ra = a[k]
    _, r0b, rb = b[k]
    d = max(abs(x - y) for x, y in zip(ra, rb))
    if d > worst:
        worst, worst_pos = d, k
    if d == 0.0:
        agree += 1
    if k[1] == 1:
        la, lb = log_softmax(r0a), log_softmax(r0b)
        kl = sum(math.exp(x) * (x - y) for x, y in zip(la, lb))
        if kl > worst_kl:
            worst_kl, worst_kl_pos = kl, k
print("bitwise-identical records: %d / %d" % (agree, len(common)))
print("max abs diff over all rows: %.6e at %s" % (worst, worst_pos))
print("max KL (T=1 row 0): %.6e at %s" % (worst_kl, worst_kl_pos))
