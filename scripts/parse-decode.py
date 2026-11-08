#!/usr/bin/env python3
# エンジンログから decode 1 リクエストぶんの (CPU ms/window, entries/layer, waitA, waitCPU, VRAM hits) を対で取り出す
import glob, re, sys, os

def parse(path):
    rows = []
    cur = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        if "strata decode timing:" in line:
            m = re.search(r"(\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window", line)
            if not m: continue
            cpu = re.search(r"CPU ([\d.]+)\]", line)
            ent = re.search(r"CPU experts ([\d.]+) \(([\d.]+) entries\)", line)
            hits = re.search(r"VRAM hits ([\d.]+)", line)
            cur = {
                "windows": int(m.group(1)), "T": float(m.group(2)), "ms_win": float(m.group(4)),
                "cpu_ms": float(cpu.group(1)) if cpu else None,
                "miss": float(ent.group(1)) if ent else None,
                "entries": float(ent.group(2)) if ent else None,
                "hits": float(hits.group(1)) if hits else None,
            }
            rows.append(cur)
        elif "strata decode GPU stages" in line and cur is not None:
            # GDN 側と QSA 側の waitA / waitCPU を足す
            wa = [float(x) for x in re.findall(r"waitA ([\d.]+)", line)]
            wc = [float(x) for x in re.findall(r"waitCPU ([\d.]+)", line)]
            cur["waitA"] = round(sum(wa), 3)
            cur["waitCPU"] = round(sum(wc), 3)
            tot = re.search(r"total ([\d.]+) ms/window over (\d+) windows", line)
            if tot: cur["gpu_total"] = float(tot.group(1))
            cur = None
    return rows

for pat in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.expanduser(pat))):
        rows = parse(f)
        print("==", os.path.basename(f))
        for i, r in enumerate(rows, 1):
            print("  #%d %5.2f ms/window  CPU %6.2f  miss %.2f  entries %.2f  hits %.2f  waitA %.2f  waitCPU %.2f"
                  % (i, r["ms_win"], r.get("cpu_ms") or 0, r.get("miss") or 0, r.get("entries") or 0,
                     r.get("hits") or 0, r.get("waitA") or 0, r.get("waitCPU") or 0))
