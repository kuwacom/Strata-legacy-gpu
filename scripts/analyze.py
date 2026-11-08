#!/usr/bin/env python3
"""Summarise an A-B-A block: per arm medians, diffs, and the engine's own read times.

Usage: analyze.py A1 B1 A2 [--kinds prose,code]
"""
import glob
import json
import os
import re
import statistics
import sys

RES = os.path.expanduser("~/work/results/pknobs/v3")


def rows(label):
    out = []
    for p in sorted(glob.glob(f"{RES}/{label}-bench.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "run" not in d:
                continue  # salt 前の attempt 1 は除外
            out.append(d)
    return out


def engine_reads(label):
    """engine log の 'prompt N tokens = R reused + K read in X ms (Y tok/s)'"""
    p = f"{RES}/{label}/state/strata/launch/docker.engine.log"
    if not os.path.exists(p):
        return []
    pat = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")
    lines = open(p, encoding="utf-8", errors="ignore").read().splitlines()
    # pristine からコピーした state には本番のログも入っている。このアームの起動以降だけ見る
    start = max([i for i, l in enumerate(lines) if "engine started:" in l] or [-1])
    out = []
    for line in lines[start + 1:]:
        m = pat.search(line)
        if m:
            out.append({"prompt": int(m.group(1)), "reused": int(m.group(2)),
                        "read": int(m.group(3)), "ms": int(m.group(4)), "tps": float(m.group(5))})
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kinds = ["prose", "code"]
    summ = {a: {} for a in args}
    for a in args:
        for r in rows(a):
            key = f"{r['kind']}-{r['target_tokens'] // 1000}k"
            summ[a].setdefault(key, {"tps": [], "ttft": [], "prompt": []})
            summ[a][key]["tps"].append(r["prefill_tps"])
            summ[a][key]["ttft"].append(r["ttft_s"])
            summ[a][key]["prompt"].append(r["prompt"])

    keys = sorted({k for a in args for k in summ[a]}, key=lambda k: (k.split("-")[0], int(k.split("-")[1][:-1])))
    print("| アーム | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for a in args:
        cells = []
        for k in keys:
            s = summ[a].get(k)
            cells.append(f"{statistics.median(s['tps']):.1f} ({len(s['tps'])}本)" if s else "-")
        print(f"| {a} | " + " | ".join(cells) + " |")

    base = args[0]
    print()
    print("| 比較 | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for a in args[1:]:
        cells = []
        for k in keys:
            if k in summ[a] and k in summ[base]:
                b = statistics.median(summ[base][k]["tps"])
                x = statistics.median(summ[a][k]["tps"])
                cells.append(f"{x:.1f} vs {b:.1f} = {(x / b - 1) * 100:+.2f}%")
            else:
                cells.append("-")
        print(f"| {a} vs {base} | " + " | ".join(cells) + " |")

    print()
    for a in args:
        er = engine_reads(a)
        if not er:
            continue
        reused = sum(1 for e in er if e["reused"])
        print(f"{a}: engine reads {len(er)}, with prefix reuse {reused}, "
              f"tps {[round(e['tps'], 1) for e in er][:12]}")


if __name__ == "__main__":
    main()
