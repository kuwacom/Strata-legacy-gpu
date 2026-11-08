import struct, sys, array

def recs(path):
    f = open(path, "rb")
    out = []
    while True:
        h = f.read(16)
        if len(h) < 16:
            break
        magic, T, nv, rec = struct.unpack("<iiii", h)
        pos, = struct.unpack("<q", f.read(8))
        buf = f.read(4 * nv)
        if len(buf) < 4 * nv:
            break
        a = array.array("f")
        a.frombytes(buf)
        am = max(range(nv), key=lambda i: a[i])
        out.append((pos, T, am, a[am]))
        f.seek(4 * nv * (T - 1), 1)
    return out

def split_requests(rs):
    reqs = []
    cur = []
    prev = None
    for r in rs:
        if r[0] == 0 and prev is not None and prev > 0:
            reqs.append(cur); cur = []
        cur.append(r); prev = r[0]
    if cur: reqs.append(cur)
    return reqs

a = recs(sys.argv[1])
b = recs(sys.argv[2])
ra = split_requests(a)
rb = split_requests(b)
print("requests A=%d B=%d, records A=%d B=%d" % (len(ra), len(rb), len(a), len(b)))
for k in range(min(len(ra), len(rb))):
    A, B = ra[k], rb[k]
    n = min(len(A), len(B))
    first_bad = None
    bad = 0
    for i in range(n):
        if A[i][0] != B[i][0] or A[i][1] != B[i][1]:
            first_bad = (i, "window", A[i][:2], B[i][:2]); break
        if A[i][2] != B[i][2]:
            bad += 1
            if first_bad is None:
                first_bad = (i, "argmax", A[i][2], B[i][2])
    print("req %2d: records %3d/%3d aligned %3d, argmax differs %d%s" % (
        k, len(A), len(B), n, bad, "" if first_bad is None else "  first: %s" % (first_bad,)))
