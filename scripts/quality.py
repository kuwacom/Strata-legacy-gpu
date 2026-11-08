"""品質の比較: 固定 4 プロンプトの回答一致（exact / 空白正規化）と、top-5 logprobs からの KL。

使い方: quality.py <results dir> <armA> <armB> [<armC> ...]
KL は各位置で両アームの top-5 の和集合に確率を寄せ直してから計算する（裾は落ちるので目安）。
出力は「A を基準にした B の平均 KL（nats, 24 token まで）」。
"""
import json
import os
import re
import sys

res = sys.argv[1]
arms = sys.argv[2:]


def load(name, arm):
    p = os.path.join(res, f"{name}-{arm}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def answers_cmp(a, b):
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            out.append("same")
        elif norm(x) == norm(y):
            out.append("ws")
        else:
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            out.append(f"diff@{j}")
    return out


def toppos(entry):
    d = {}
    for cand in entry.get("top_logprobs") or []:
        d[cand["token"]] = cand["logprob"]
    return d


def kl_pos(pa, pb):
    keys = set(pa) | set(pb)
    if not keys:
        return None
    # 共通サポートに寄せ直す（top-5 の外は等しく落ちるとみなす）
    import math
    za = sum(math.exp(pa.get(k, -99.0)) for k in keys)
    zb = sum(math.exp(pb.get(k, -99.0)) for k in keys)
    if za <= 0 or zb <= 0:
        return None
    kl = 0.0
    for k in keys:
        pa_k = math.exp(pa.get(k, -99.0)) / za
        pb_k = math.exp(pb.get(k, -99.0)) / zb
        if pa_k > 0:
            kl += pa_k * math.log(pa_k / max(pb_k, 1e-12))
    return kl


def kl_cmp(a, b):
    kls = []
    for ea, eb in zip(a, b):
        for ta, tb in zip(ea.get("tokens") or [], eb.get("tokens") or []):
            v = kl_pos(toppos(ta), toppos(tb))
            if v is not None:
                kls.append(v)
    return kls


if len(arms) < 2:
    raise SystemExit("need at least two arms")
ref = arms[0]
ans = {a: load("answers", a) for a in arms}
kl = {a: load("kl", a) for a in arms}
print(f"ref={ref}")
for a in arms[1:]:
    if ans.get(ref) and ans.get(a):
        print(f"answers {ref} vs {a}: {answers_cmp(ans[ref], ans[a])}")
    else:
        print(f"answers {ref} vs {a}: missing")
    if kl.get(ref) and kl.get(a):
        k = kl_cmp(kl[ref], kl[a])
        if k:
            print(f"KL {ref} -> {a}: mean {sum(k)/len(k):.5f} nats over {len(k)} positions, max {max(k):.5f}")
        else:
            print(f"KL {ref} -> {a}: no logprob positions")
    else:
        print(f"KL {ref} -> {a}: missing")
