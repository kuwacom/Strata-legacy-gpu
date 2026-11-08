"""v3b 品質比較: answers4 / answers / klprobe / qa の結果を突き合わせる。

使い方: qualityCmp.py <pknobs|dknobs> <results dir> <args...>
  pknobs: qualityCmp.py pknobs <dir> <armA> <armB>   # ans4-<arm>-pass*.json と qa/needle を見る
  dknobs: qualityCmp.py dknobs <dir> <armA> <armB>   # answers-<arm>-pass*.json と kl-<arm>-pass*.json を見る
"""
import difflib
import json
import math
import os
import sys

kind, res = sys.argv[1], sys.argv[2]


def load(name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8").read().strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def textdiff(a, b):
    same = a == b
    n = sum(1 for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
    return same, n


if kind == "pknobs":
    armA, armB = sys.argv[3], sys.argv[4]

    def texts_of(arm):
        out = []
        for f in [f"ans4-{arm}-pass1.json", f"ans4-{arm}-pass2.json"]:
            d = load(f)
            if d:
                out.extend(r.get("answer", "") if isinstance(r, dict) else r for r in (d if isinstance(d, list) else [d]))
        return out

    texts = {armA: texts_of(armA), armB: texts_of(armB)}
    print("== 固定 4 プロンプト（24K、temp 0）の回答一致 ==")
    nmax = min(len(texts[armA]), len(texts[armB]))
    for i in range(min(nmax, 4)):
        same, n = textdiff(texts[armA][i], texts[armB][i])
        print(f"  prompt {i+1}: A vs B bytes-equal={same} diff-lines={n} lenA={len(texts[armA][i])} lenB={len(texts[armB][i])}")
    if len(texts[armA]) >= 8:
        print("  -- A 自己一致（pass1 vs pass2）--")
        for i in range(4):
            same, n = textdiff(texts[armA][i], texts[armA][4 + i])
            print(f"  prompt {i+1}: bytes-equal={same} diff-lines={n}")
    for arm in (armA, armB):
        for f in [f"needle-{arm}-128000.jsonl", f"needle-{arm}-480000.jsonl"]:
            p = os.path.join(res, f)
            if os.path.exists(p):
                for line in open(p):
                    d = json.loads(line)
                    print(f"  needle {arm} {int(d['target_tokens'])//1000}k {d['pos_pct']:.0f}%: found={d['found']} prompt={d['prompt']} ttft={d['ttft_s']}s")
        q = load(f"qa-{arm}.json")
        if q:
            print(f"  qa {arm}: score={q.get('score')}/{q.get('total')} wall={q.get('wall_s')}s")
elif kind == "dknobs":
    armA, armB = sys.argv[3], sys.argv[4]
    print("== 固定 4 プロンプトの greedy 回答一致 ==")
    tA, tB = [], []
    for f in [f"answers-{armA}-pass1.json", f"answers-{armA}-pass2.json"]:
        d = load(f)
        tA.extend(d or [])
    for f in [f"answers-{armB}-pass1.json"]:
        d = load(f)
        tB.extend(d or [])
    for i in range(min(len(tA), len(tB))):
        same, n = textdiff(tA[i], tB[i])
        print(f"  prompt {i+1}: A vs B bytes-equal={same} diff-lines={n} lenA={len(tA[i])} lenB={len(tB[i])}")
    if len(tA) > 4:
        print("  -- A 自己一致（pass1 vs pass2）--")
        for i in range(4):
            same, n = textdiff(tA[i], tA[4 + i])
            print(f"  prompt {i+1}: bytes-equal={same} diff-lines={n}")

    print("== logprobs 一致（greedy 24 token の top-5）==")
    kl = {}
    for arm in (armA, armB):
        rows = []
        for f in [f"kl-{arm}-pass1.json", f"kl-{arm}-pass2.json"]:
            d = load(f)
            rows.extend(d or [])
        kl[arm] = rows
    if kl[armA] and kl[armB]:
        a = [(o["content"], o["tokens"]) for o in kl[armA]]
        b = [(o["content"], o["tokens"]) for o in kl[armB]]
        n = min(len(a), len(b))
        same_content = sum(1 for i in range(n) if a[i][0] == b[i][0])
        tot = agree = 0
        kldiv = []
        for i in range(n):
            ta, tb = a[i][1], b[i][1]
            m = min(len(ta), len(tb))
            for j in range(m):
                la, lb = ta[j], tb[j]
                tot += 1
                top_a = la["top_logprobs"][0]["token"] if la.get("top_logprobs") else None
                top_b = lb["top_logprobs"][0]["token"] if lb.get("top_logprobs") else None
                agree += int(top_a == top_b)
                if top_a == top_b and la.get("top_logprobs") and lb.get("top_logprobs"):
                    pa = math.exp(la["top_logprobs"][0]["logprob"])
                    pb = math.exp(lb["top_logprobs"][0]["logprob"])
                    if pa > 0 and pb > 0:
                        kldiv.append(abs(math.log(pa) - math.log(pb)))
        print(f"  greedy 内容一致: {same_content}/{n} prompts")
        print(f"  top-1 token 一致率: {agree}/{tot} = {agree / tot * 100:.2f}%" if tot else "  (logprobs なし)")
        if kldiv:
            kldiv.sort()
            print(f"  top-1 logprob 差: 中央値 {kldiv[len(kldiv)//2]:.4f} 最大 {kldiv[-1]:.4f} (n={len(kldiv)})")
    else:
        print("  (kl ファイルなし)")
