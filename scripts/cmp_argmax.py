import struct, sys

def rows(path, upto):
    f = open(path, "rb")
    out = []
    while len(out) < upto:
        h = f.read(16)
        if len(h) < 16:
            break
        magic, T, nv, rec = struct.unpack("<iiii", h)
        pos, = struct.unpack("<q", f.read(8))
        n = T * nv
        data = f.read(4 * n)
        if len(data) < 4 * n:
            break
        row0 = struct.unpack("<%df" % nv, data[:4 * nv])
        am = max(range(nv), key=lambda i: row0[i])
        srt = sorted(row0, reverse=True)
        out.append((pos, T, am, srt[0], srt[1]))
    return out

a = rows(sys.argv[1], int(sys.argv[3]))
b = rows(sys.argv[2], int(sys.argv[3]))
n = min(len(a), len(b))
print("rec  pos    T  A(argmax, top, second)            B(argmax, top, second)            same")
bad = 0
for i in range(n):
    pa, Ta, ama, ta, sa = a[i]
    pb, Tb, amb, tb, sb = b[i]
    same = "yes" if (ama == amb and Ta == Tb and pa == pb) else "NO"
    if same == "NO":
        bad += 1
    print("%3d %6d %2d  %8d %.4f %.4f   %8d %.4f %.4f   %s" % (i, pa, Ta, ama, ta, sa, amb, tb, sb, same))
print("argmax differs in %d/%d records" % (bad, n))
