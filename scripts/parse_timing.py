#!/usr/bin/env python3
# prefill timing 行から phase 内訳とチャンク長を抜く
# 使い方: parse_timing.py <engine.log> [label]
import re, sys, collections

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
chunks = []
rows = []
for ln in lines:
    m = re.search(r"prompt chunk auto: (\d+) tokens", ln)
    if m:
        chunks.append(int(m.group(1)))
    if "strata prefill timing:" in ln and " tokens," in ln:
        mt = re.search(r"prefill timing: (\d+) tokens, GPU timeline ([\d.]+) ms, wall ([\d.]+) ms", ln)
        if not mt:
            continue
        phases = {}
        for pm in re.finditer(r"([a-z][a-z0-9 +/:_-]*?) (\d+) \((\d+\.\d)%\)", ln):
            phases[pm.group(1).strip()] = (int(pm.group(2)), float(pm.group(3)))
        rows.append((int(mt.group(1)), float(mt.group(2)), float(mt.group(3)), phases))

print(f"# {label}")
if chunks:
    print("chunks:", chunks)
for toks, gpu, wall, ph in rows:
    gdn = ph.get("gdn", (0, 0.0))
    out = ph.get("gdn out proj", (0, 0.0))
    conv = ph.get("gdn conv+gates", (0, 0.0))
    rec = ph.get("gdn recurrence", (0, 0.0))
    qsa = ph.get("qsa proj", (0, 0.0))
    print(f"T={toks} gpu={gpu:.0f} wall={wall:.0f} | gdn {gdn[0]}ms {gdn[1]}% | out {out[0]}ms {out[1]}% | conv {conv[0]} | rec {rec[0]} | qsa proj {qsa[0]}ms {qsa[1]}%")
    top = sorted(ph.items(), key=lambda kv: -kv[1][0])[:6]
    print("  top:", " ".join(f"{k} {v[0]}ms({v[1]}%)" for k, v in top))
