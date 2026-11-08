import struct, sys

def read(path):
    recs = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            magic, T, nv, rec = struct.unpack("<iiii", hdr)
            pos, = struct.unpack("<q", f.read(8))
            n = T * nv
            data = f.read(4 * n)
            if len(data) < 4 * n:
                break
            recs.append((magic, T, nv, rec, pos, data))
    return recs

a = read(sys.argv[1])
b = read(sys.argv[2])
print("records A=%d B=%d" % (len(a), len(b)))
if len(a) != len(b):
    print("RECORD_COUNT_DIFF")
n = min(len(a), len(b))
bad = 0
maxabs = 0.0
for i in range(n):
    ma, Ta, nva, ra, pa, da = a[i]
    mb, Tb, nvb, rb, pb, db = b[i]
    if (Ta, nva, pa) != (Tb, nvb, pb):
        print("rec %d header differs: %s vs %s" % (i, (Ta, nva, pa), (Tb, nvb, pb)))
        bad += 1
        continue
    if da == db:
        continue
    bad += 1
    fa = struct.unpack("<%df" % (Ta * nva), da)
    fb = struct.unpack("<%df" % (Tb * nvb), db)
    nd = 0
    m = 0.0
    first = None
    for k, (x, y) in enumerate(zip(fa, fb)):
        if x != y:
            nd += 1
            if first is None:
                first = (k, x, y)
            m = max(m, abs(x - y))
    maxabs = max(maxabs, m)
    print("rec %d pos %d T %d: %d/%d floats differ, max |d| %.6g, first %s" % (i, pa, Ta, nd, len(fa), m, first))
print("records differing: %d/%d, overall max |d| %.6g" % (bad, n, maxabs))
print("BITWISE_IDENTICAL" if bad == 0 and len(a) == len(b) else "NOT_IDENTICAL")
